//go:build mlx

package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type stubCache struct {
	freeCalls int
}

func (s *stubCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) { return keys, values }
func (s *stubCache) State() (*mlx.Array, *mlx.Array)                         { return nil, nil }
func (s *stubCache) Materialize() []*mlx.Array                               { return nil }
func (s *stubCache) CanTrim() bool                                           { return true }
func (s *stubCache) Trim(int) int                                            { return 0 }
func (s *stubCache) Clone() cache.Cache                                      { return s }
func (s *stubCache) Free()                                                   { s.freeCalls++ }
func (s *stubCache) Offset() int                                             { return 0 }
func (s *stubCache) Len() int                                                { return 0 }

func TestPrefillChunkSize(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PREFILL_CHUNK", "")
	if got := prefillChunkSize(false); got != 2<<10 {
		t.Fatalf("prefillChunkSize(false) = %d, want %d", got, 2<<10)
	}
	if got := prefillChunkSize(true); got != 32 {
		t.Fatalf("prefillChunkSize(true) = %d, want %d", got, 32)
	}
}

func TestPrefillChunkSizeEnvOverride(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PREFILL_CHUNK", "96")
	if got := prefillChunkSize(false); got != 96 {
		t.Fatalf("prefillChunkSize(false) with env = %d, want %d", got, 96)
	}
	if got := prefillChunkSize(true); got != 96 {
		t.Fatalf("prefillChunkSize(true) with env = %d, want %d", got, 96)
	}
}

func TestMLXDebugMemoryEnabled(t *testing.T) {
	t.Setenv("OLLAMA_MLX_DEBUG_MEMORY", "")
	if mlxDebugMemoryEnabled() {
		t.Fatal("mlxDebugMemoryEnabled() = true, want false")
	}

	t.Setenv("OLLAMA_MLX_DEBUG_MEMORY", "1")
	if !mlxDebugMemoryEnabled() {
		t.Fatal("mlxDebugMemoryEnabled() = false, want true")
	}
}

func TestHasRecurrentCaches(t *testing.T) {
	if hasRecurrentCaches(nil) {
		t.Fatal("hasRecurrentCaches(nil) = true, want false")
	}

	if hasRecurrentCaches([]cache.Cache{cache.NewKVCache()}) {
		t.Fatal("hasRecurrentCaches(kv-only) = true, want false")
	}

	rc := cache.NewRecurrentCache(4, 8, 2, 16, 8)
	if !hasRecurrentCaches([]cache.Cache{cache.NewKVCache(), rc}) {
		t.Fatal("hasRecurrentCaches(mixed) = false, want true")
	}
}

func TestRecurrentMaterializeInterval(t *testing.T) {
	t.Setenv("OLLAMA_MLX_RECURRENT_MATERIALIZE_INTERVAL", "")

	if got := recurrentMaterializeInterval(true, true); got != 0 {
		t.Fatalf("recurrentMaterializeInterval(lowmem=true, recurrent=true) = %d, want 0", got)
	}
	if got := recurrentMaterializeInterval(false, false); got != 0 {
		t.Fatalf("recurrentMaterializeInterval(lowmem=false, recurrent=false) = %d, want 0", got)
	}
	if got := recurrentMaterializeInterval(false, true); got != defaultRecurrentMaterializeInterval {
		t.Fatalf("recurrentMaterializeInterval(default) = %d, want %d", got, defaultRecurrentMaterializeInterval)
	}

	t.Setenv("OLLAMA_MLX_RECURRENT_MATERIALIZE_INTERVAL", "16")
	if got := recurrentMaterializeInterval(false, true); got != 16 {
		t.Fatalf("recurrentMaterializeInterval(env=16) = %d, want 16", got)
	}

	t.Setenv("OLLAMA_MLX_RECURRENT_MATERIALIZE_INTERVAL", "0")
	if got := recurrentMaterializeInterval(false, true); got != 0 {
		t.Fatalf("recurrentMaterializeInterval(env=0) = %d, want 0", got)
	}

	t.Setenv("OLLAMA_MLX_RECURRENT_MATERIALIZE_INTERVAL", "-1")
	if got := recurrentMaterializeInterval(false, true); got != 0 {
		t.Fatalf("recurrentMaterializeInterval(env=-1) = %d, want 0", got)
	}
}

func TestMLXPipelineTimingConfig(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PIPELINE_TIMING", "")
	t.Setenv("OLLAMA_MLX_PIPELINE_TIMING_EVERY", "")
	if enabled, every := mlxPipelineTimingConfig(); enabled || every != 0 {
		t.Fatalf("mlxPipelineTimingConfig() = (%v, %d), want (false, 0)", enabled, every)
	}

	t.Setenv("OLLAMA_MLX_PIPELINE_TIMING", "1")
	if enabled, every := mlxPipelineTimingConfig(); !enabled || every != defaultPipelineTimingEvery {
		t.Fatalf("mlxPipelineTimingConfig(enabled default) = (%v, %d), want (true, %d)", enabled, every, defaultPipelineTimingEvery)
	}

	t.Setenv("OLLAMA_MLX_PIPELINE_TIMING_EVERY", "16")
	if enabled, every := mlxPipelineTimingConfig(); !enabled || every != 16 {
		t.Fatalf("mlxPipelineTimingConfig(enabled env=16) = (%v, %d), want (true, 16)", enabled, every)
	}

	t.Setenv("OLLAMA_MLX_PIPELINE_TIMING_EVERY", "0")
	if enabled, every := mlxPipelineTimingConfig(); !enabled || every != defaultPipelineTimingEvery {
		t.Fatalf("mlxPipelineTimingConfig(enabled env=0) = (%v, %d), want (true, %d)", enabled, every, defaultPipelineTimingEvery)
	}

	t.Setenv("OLLAMA_MLX_PIPELINE_TIMING", "0")
	if enabled, every := mlxPipelineTimingConfig(); enabled || every != 0 {
		t.Fatalf("mlxPipelineTimingConfig(disabled) = (%v, %d), want (false, 0)", enabled, every)
	}
}

func TestMLXComputeLogprobsEnabled(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PIPELINE_COMPUTE_LOGPROBS", "")
	if mlxComputeLogprobsEnabled() {
		t.Fatal("mlxComputeLogprobsEnabled() = true, want false")
	}

	t.Setenv("OLLAMA_MLX_PIPELINE_COMPUTE_LOGPROBS", "1")
	if !mlxComputeLogprobsEnabled() {
		t.Fatal("mlxComputeLogprobsEnabled() = false with env=1, want true")
	}

	t.Setenv("OLLAMA_MLX_PIPELINE_COMPUTE_LOGPROBS", "0")
	if mlxComputeLogprobsEnabled() {
		t.Fatal("mlxComputeLogprobsEnabled() = true with env=0, want false")
	}
}

func TestFinalizeRequestCachesUsesPromptCachePath(t *testing.T) {
	insertCalls := 0
	freeCalls := 0
	logPhase := ""

	finalizeRequestCaches(
		true,
		func() { insertCalls++ },
		func() { freeCalls++ },
		func(phase string, _ int) { logPhase = phase },
	)

	if insertCalls != 1 {
		t.Fatalf("insert calls = %d, want 1", insertCalls)
	}
	if freeCalls != 0 {
		t.Fatalf("free calls = %d, want 0", freeCalls)
	}
	if logPhase != "request_done_cached" {
		t.Fatalf("log phase = %q, want %q", logPhase, "request_done_cached")
	}
}

func TestFinalizeRequestCachesUsesFreePath(t *testing.T) {
	insertCalls := 0
	freeCalls := 0
	logPhase := ""

	finalizeRequestCaches(
		false,
		func() { insertCalls++ },
		func() { freeCalls++ },
		func(phase string, _ int) { logPhase = phase },
	)

	if insertCalls != 0 {
		t.Fatalf("insert calls = %d, want 0", insertCalls)
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
	if logPhase != "request_done_freed" {
		t.Fatalf("log phase = %q, want %q", logPhase, "request_done_freed")
	}
}

func TestFreeOwnedCaches(t *testing.T) {
	a := &stubCache{}
	b := &stubCache{}
	caches := []cache.Cache{a, nil, b}

	freeOwnedCaches(caches)

	if a.freeCalls != 1 {
		t.Fatalf("a free calls = %d, want 1", a.freeCalls)
	}
	if b.freeCalls != 1 {
		t.Fatalf("b free calls = %d, want 1", b.freeCalls)
	}
	if caches[0] != nil || caches[2] != nil {
		t.Fatalf("cache entries not nilled after free: %#v", caches)
	}
}
