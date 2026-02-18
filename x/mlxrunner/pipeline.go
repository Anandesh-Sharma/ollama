//go:build mlx

package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const defaultRecurrentMaterializeInterval = 64
const defaultPipelineTimingEvery = 64

func prefillChunkSize(lowMemoryDecode bool) int {
	if v := os.Getenv("OLLAMA_MLX_PREFILL_CHUNK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}

	if lowMemoryDecode {
		// Recurrent/no-prompt-cache path favors lower peak memory over prefill throughput.
		// Keep this conservative to avoid transient prefill spikes and allocator thrash.
		return 32
	}
	return 2 << 10
}

func hasRecurrentCaches(caches []cache.Cache) bool {
	for _, c := range caches {
		if c == nil {
			continue
		}
		if _, ok := c.(*cache.RecurrentCache); ok {
			return true
		}
	}
	return false
}

// recurrentMaterializeInterval controls periodic recurrent-cache materialization
// during async decode. It exists to bound graph/handle growth when using fast
// recurrent cache writes; it is primarily a memory/stability tuning knob, not a
// throughput knob.
func recurrentMaterializeInterval(lowMemoryDecode bool, hasRecurrent bool) int {
	if lowMemoryDecode || !hasRecurrent {
		return 0
	}
	if v := os.Getenv("OLLAMA_MLX_RECURRENT_MATERIALIZE_INTERVAL"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			if n < 0 {
				return 0
			}
			return n
		}
	}
	return defaultRecurrentMaterializeInterval
}

func mlxDebugMemoryEnabled() bool {
	return os.Getenv("OLLAMA_MLX_DEBUG_MEMORY") != ""
}

// mlxPipelineTimingConfig controls runner-side decode pipeline timing logs. This
// is diagnostic-only and intentionally separate from model-specific timing.
func mlxPipelineTimingConfig() (enabled bool, every int) {
	if v, ok := os.LookupEnv("OLLAMA_MLX_PIPELINE_TIMING"); ok {
		if parsed, err := strconv.ParseBool(v); err == nil {
			enabled = parsed
		}
	}
	if !enabled {
		return false, 0
	}
	every = defaultPipelineTimingEvery
	if v := os.Getenv("OLLAMA_MLX_PIPELINE_TIMING_EVERY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			every = n
		}
	}
	return true, every
}

// mlxComputeLogprobsEnabled restores the old decode-step logprob normalization
// path for profiling/experiments. It is off by default because the MLX runner
// does not currently populate Response.Logprobs.
func mlxComputeLogprobsEnabled() bool {
	if v, ok := os.LookupEnv("OLLAMA_MLX_PIPELINE_COMPUTE_LOGPROBS"); ok {
		if enabled, err := strconv.ParseBool(v); err == nil {
			return enabled
		}
	}
	// The MLX runner currently does not populate Response.Logprobs, so skip the
	// full-vocab logprob normalization path unless explicitly requested for
	// debugging/experiments.
	return false
}

type pipelineTiming struct {
	every int

	stepCalls  int
	stepAsync  int
	stepSync   int
	sampleInts int

	stepTotalDur  time.Duration
	forwardDur    time.Duration
	unembedDur    time.Duration
	sliceDur      time.Duration
	logprobsDur   time.Duration
	sampleDur     time.Duration
	pinSweepDur   time.Duration
	asyncEvalDur  time.Duration
	sampleIntDur  time.Duration
	lastEmitCount int
}

func newPipelineTiming() *pipelineTiming {
	enabled, every := mlxPipelineTimingConfig()
	if !enabled {
		return nil
	}
	pt := &pipelineTiming{every: every}
	fmt.Fprintf(os.Stderr, "mlx pipeline timing: enabled every=%d\n", every)
	return pt
}

func (pt *pipelineTiming) recordStep(
	async bool,
	total, forward, unembed, slice, logprobs, sample, pinSweep, asyncEval time.Duration,
) {
	if pt == nil {
		return
	}
	pt.stepCalls++
	if async {
		pt.stepAsync++
	} else {
		pt.stepSync++
	}
	pt.stepTotalDur += total
	pt.forwardDur += forward
	pt.unembedDur += unembed
	pt.sliceDur += slice
	pt.logprobsDur += logprobs
	pt.sampleDur += sample
	pt.pinSweepDur += pinSweep
	pt.asyncEvalDur += asyncEval
}

func (pt *pipelineTiming) recordSampleInt(d time.Duration, decodeCount int) {
	if pt == nil {
		return
	}
	pt.sampleInts++
	pt.sampleIntDur += d
	pt.maybeEmit(false, decodeCount)
}

func (pt *pipelineTiming) maybeEmit(force bool, decodeCount int) {
	if pt == nil {
		return
	}
	if !force {
		if pt.every <= 0 || decodeCount <= 0 || decodeCount%pt.every != 0 {
			return
		}
	}
	if pt.lastEmitCount == decodeCount {
		return
	}
	pt.lastEmitCount = decodeCount

	msAvg := func(d time.Duration, n int) float64 {
		if n <= 0 {
			return 0
		}
		return float64(d) / float64(n) / float64(time.Millisecond)
	}
	stepResidual := pt.stepTotalDur - pt.forwardDur - pt.unembedDur - pt.sliceDur - pt.logprobsDur - pt.sampleDur - pt.pinSweepDur - pt.asyncEvalDur
	if stepResidual < 0 {
		stepResidual = 0
	}
	fmt.Fprintf(
		os.Stderr,
		"mlx pipeline timing: decode=%d step_calls=%d step_async=%d step_sync=%d avg_step_ms=%.2f fwd_ms=%.2f unembed_ms=%.2f slice_ms=%.2f logprobs_ms=%.2f sample_ms=%.2f pin_sweep_ms=%.2f async_eval_ms=%.2f step_residual_ms=%.2f sample_int_ms=%.2f\n",
		decodeCount,
		pt.stepCalls,
		pt.stepAsync,
		pt.stepSync,
		msAvg(pt.stepTotalDur, pt.stepCalls),
		msAvg(pt.forwardDur, pt.stepCalls),
		msAvg(pt.unembedDur, pt.stepCalls),
		msAvg(pt.sliceDur, pt.stepCalls),
		msAvg(pt.logprobsDur, pt.stepCalls),
		msAvg(pt.sampleDur, pt.stepCalls),
		msAvg(pt.pinSweepDur, pt.stepCalls),
		msAvg(pt.asyncEvalDur, pt.stepCalls),
		msAvg(stepResidual, pt.stepCalls),
		msAvg(pt.sampleIntDur, pt.sampleInts),
	)
}

func finalizeRequestCaches(usePromptCache bool, insertCache func(), freeCaches func(), logMemory func(string, int)) {
	if usePromptCache {
		insertCache()
		logMemory("request_done_cached", -1)
		return
	}
	freeCaches()
	logMemory("request_done_freed", -1)
}

func recordCacheCheckpoints(caches []cache.Cache, pos int) {
	if pos <= 0 {
		return
	}
	for _, c := range caches {
		if c == nil {
			continue
		}
		if recorder, ok := c.(cache.CheckpointRecorder); ok {
			recorder.RecordCheckpoint(pos)
		}
	}
}

func freeOwnedCaches(caches []cache.Cache) {
	for i, c := range caches {
		if c == nil {
			continue
		}
		c.Free()
		caches[i] = nil
	}
}

func (r *Runner) TextGenerationPipeline(request Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	enableCompile := true
	if modelCompile, ok := r.Model.(interface{ EnableCompile() bool }); ok {
		enableCompile = modelCompile.EnableCompile()
	}
	if enableCompile {
		mlx.EnableCompile()
	} else {
		mlx.DisableCompile()
	}

	inputs := r.Tokenizer.Encode(request.Prompt, true)

	usePromptCache := true
	if m, ok := r.Model.(interface{ DisablePromptCache() bool }); ok && m.DisablePromptCache() {
		usePromptCache = false
	}
	lowMemoryDecode := !usePromptCache
	if m, ok := r.Model.(interface{ LowMemoryDecode() bool }); ok {
		lowMemoryDecode = m.LowMemoryDecode()
	}
	prefillChunk := prefillChunkSize(lowMemoryDecode)

	var caches []cache.Cache
	var tokens []int32
	if usePromptCache {
		caches, tokens = r.FindNearestCache(inputs)
	} else {
		tokens = inputs
	}

	if len(caches) == 0 {
		if cacheFactory, ok := r.Model.(interface{ NewCaches() []cache.Cache }); ok {
			caches = cacheFactory.NewCaches()
		} else {
			caches = make([]cache.Cache, r.Model.NumLayers())
			for i := range caches {
				caches[i] = cache.NewKVCache()
			}
		}
	}

	materializeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = append(state, c.Materialize()...)
		}
		if len(state) == 0 {
			return
		}
		mlx.Eval(state...)
	}
	materializeRecurrentCaches := func() bool {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			if c == nil {
				continue
			}
			if _, ok := c.(*cache.RecurrentCache); !ok {
				continue
			}
			state = append(state, c.Materialize()...)
		}
		if len(state) == 0 {
			return false
		}
		mlx.Eval(state...)
		return true
	}
	freeCaches := func() {
		// Non-prompt-cache requests allocate fresh caches every generation.
		// Explicitly free cache-owned state (including recurrent checkpoints),
		// then sweep remaining intermediates.
		freeOwnedCaches(caches)
		mlx.Sweep()
		mlx.ClearCache()
	}
	debugMemory := mlxDebugMemoryEnabled()
	hasRecurrent := hasRecurrentCaches(caches)
	asyncRecurrentMaterializeEvery := recurrentMaterializeInterval(lowMemoryDecode, hasRecurrent)
	computeStepLogprobs := mlxComputeLogprobsEnabled()
	pipelineTiming := newPipelineTiming()
	logMemory := func(phase string, token int) {
		if !debugMemory {
			return
		}
		if token >= 0 {
			slog.Info("MLX memory", "phase", phase, "token", token, "memory", mlx.Memory{})
			return
		}
		slog.Info("MLX memory", "phase", phase, "memory", mlx.Memory{})
	}
	logMemory("prefill_start", -1)

	total, processed := len(tokens), 0
	slog.Info("Prompt processing progress", "processed", processed, "total", total)
	for total-processed > 1 {
		n := min(prefillChunk, total-processed-1)
		r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		mlx.Sweep()
		materializeCaches()
		recordCacheCheckpoints(caches, processed+n)
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		mlx.ClearCache()
	}
	logMemory("prefill_done", -1)

	step := func(token *mlx.Array, async bool) (*mlx.Array, *mlx.Array) {
		var t0, t time.Time
		var forwardDur, unembedDur, sliceDur, logprobsDur, sampleDur, pinSweepDur, asyncEvalDur time.Duration
		if pipelineTiming != nil {
			t0 = time.Now()
			t = t0
		}

		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		if pipelineTiming != nil {
			forwardDur = time.Since(t)
			t = time.Now()
		}

		logits := r.Model.Unembed(fwd)
		if pipelineTiming != nil {
			unembedDur = time.Since(t)
			t = time.Now()
		}

		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)
		if pipelineTiming != nil {
			sliceDur = time.Since(t)
			t = time.Now()
		}

		var logprobs *mlx.Array
		sampleInput := logits
		if computeStepLogprobs {
			logprobs = logits.Subtract(logits.Logsumexp(true))
			sampleInput = logprobs
		}
		if pipelineTiming != nil {
			logprobsDur = time.Since(t)
			t = time.Now()
		}

		sample := request.Sample(sampleInput)
		if pipelineTiming != nil {
			sampleDur = time.Since(t)
			t = time.Now()
		}

		mlx.Pin(sample, logprobs)
		mlx.Sweep()
		if pipelineTiming != nil {
			pinSweepDur = time.Since(t)
		}
		if async {
			mlx.AsyncEval(sample, logprobs)
			if pipelineTiming != nil {
				asyncEvalDur = time.Since(t)
			}
		}
		if pipelineTiming != nil {
			pipelineTiming.recordStep(async, time.Since(t0), forwardDur, unembedDur, sliceDur, logprobsDur, sampleDur, pinSweepDur, asyncEvalDur)
		}

		return sample, logprobs
	}

	sample, logprobs := step(mlx.FromValues(tokens[processed:], total-processed), !lowMemoryDecode)
	if lowMemoryDecode {
		// Materialize cache updates to prevent transform graph growth.
		materializeCaches()
	}
	recordCacheCheckpoints(caches, total)
	logMemory("decode_init", -1)

	var b bytes.Buffer

	now := time.Now()
	final := Response{Done: true, PromptTokens: total, CompletionTokens: request.Options.MaxTokens, DoneReason: 1}
	outputs := make([]int32, 0, request.Options.MaxTokens)
	for i := range request.Options.MaxTokens {
		var nextSample, nextLogprobs *mlx.Array
		if !lowMemoryDecode {
			nextSample, nextLogprobs = step(sample, true)
		}
		if i == 0 {
			slog.Info("Prompt processing progress", "processed", total, "total", total)
			mlx.Eval(sample)
			logMemory("decode_first_eval", i)
			final.PromptTokensDuration = time.Since(now)
			now = time.Now()
		}

		var intWaitStart time.Time
		if pipelineTiming != nil {
			intWaitStart = time.Now()
		}
		output := int32(sample.Int())
		if pipelineTiming != nil {
			pipelineTiming.recordSampleInt(time.Since(intWaitStart), len(outputs)+1)
		}
		outputs = append(outputs, output)
		if !lowMemoryDecode {
			recordCacheCheckpoints(caches, total+len(outputs))
		}

		if r.Tokenizer.IsEOS(output) {
			mlx.Unpin(nextSample, nextLogprobs)
			mlx.Unpin(sample, logprobs)
			final.Token = int(output)
			final.DoneReason = 0
			final.CompletionTokens = i
			break
		}

		request.Responses <- Response{
			Text:  r.Decode(output, &b),
			Token: int(output),
		}

		// For recurrent linear-attention models, avoid async prefetch to reduce
		// peak memory and clear allocator cache every token.
		if lowMemoryDecode {
			mlx.Unpin(sample, logprobs)
			mlx.Sweep()
			if i+1 >= request.Options.MaxTokens {
				break
			}
			mlx.ClearCache()
			sample, logprobs = step(mlx.FromValues([]int32{output}, 1), false)
			// Materialize cache updates to avoid unbounded transform chains.
			materializeCaches()
			recordCacheCheckpoints(caches, total+len(outputs))
			if i%32 == 0 {
				logMemory("decode_lowmem_step", i)
			}
			continue
		}

		mlx.Unpin(sample, logprobs)
		if asyncRecurrentMaterializeEvery > 0 && (i+1)%asyncRecurrentMaterializeEvery == 0 {
			if materializeRecurrentCaches() {
				mlx.Sweep()
				logMemory("decode_async_recurrent_materialize", i)
			}
		}
		if i%256 == 0 {
			mlx.ClearCache()
		}
		if i%64 == 0 {
			logMemory("decode_async_step", i)
		}

		sample, logprobs = nextSample, nextLogprobs
	}

	mlx.Unpin(sample, logprobs)
	if pipelineTiming != nil {
		pipelineTiming.maybeEmit(true, len(outputs))
	}
	final.CompletionTokensDuration = time.Since(now)
	request.Responses <- final
	finalizeRequestCaches(usePromptCache,
		func() { r.InsertCache(append(inputs, outputs...), caches) },
		freeCaches,
		logMemory,
	)
	mlx.Sweep()

	if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
		mlx.LogArrays()
		if r.cache != nil {
			r.cache.LogCache()
		}
	}
	return nil
}

func (r Runner) Decode(sample int32, b *bytes.Buffer) string {
	token := r.Tokenizer.Decode([]int32{sample})

	if _, err := b.WriteString(token); err != nil {
		slog.Error("Failed to write token to buffer", "error", err)
		return ""
	}

	return flushValidUTF8Prefix(b)
}
