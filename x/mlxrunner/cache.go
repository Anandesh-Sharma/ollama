//go:build mlx

package mlxrunner

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const defaultPromptCacheBranches = 1

// HybridCacheEntry stores a single prompt branch with mixed cache types
// (e.g. KV + recurrent caches) and coordinates shared operations across them.
type HybridCacheEntry struct {
	Tokens []int32
	Caches []cache.Cache
}

// CacheEntry is kept as an alias for the current single-entry runner path.
// Future multi-entry cache stores should prefer HybridCacheEntry directly.
type CacheEntry = HybridCacheEntry

func promptCacheBranchLimit() int {
	if v := os.Getenv("OLLAMA_MLX_PROMPT_CACHE_BRANCHES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultPromptCacheBranches
}

func cloneTokens(tokens []int32) []int32 {
	out := make([]int32, len(tokens))
	copy(out, tokens)
	return out
}

func equalTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (r *Runner) cacheStore() []*HybridCacheEntry {
	if len(r.caches) == 0 && r.cache != nil {
		r.caches = []*HybridCacheEntry{r.cache}
	}
	if r.cache == nil && len(r.caches) > 0 {
		r.cache = r.caches[0]
	}
	return r.caches
}

func (r *Runner) setCacheStore(entries []*HybridCacheEntry) {
	r.caches = entries
	if len(entries) == 0 {
		r.cache = nil
		return
	}
	r.cache = entries[0]
}

func (r *Runner) touchCacheEntry(idx int) {
	if idx <= 0 || idx >= len(r.caches) {
		return
	}
	e := r.caches[idx]
	copy(r.caches[1:idx+1], r.caches[:idx])
	r.caches[0] = e
	r.cache = r.caches[0]
}

func (r *Runner) bestCacheEntry(tokens []int32) (idx int, prefix int) {
	bestIdx, bestPrefix := -1, 0
	for i, e := range r.cacheStore() {
		if e == nil {
			continue
		}
		p := e.PrefixLen(tokens)
		if p > bestPrefix {
			bestIdx, bestPrefix = i, p
		}
	}
	return bestIdx, bestPrefix
}

func (e *HybridCacheEntry) PrefixLen(tokens []int32) int {
	if e == nil {
		return 0
	}
	prefix := 0
	for prefix < len(tokens) && prefix < len(e.Tokens) && tokens[prefix] == e.Tokens[prefix] {
		prefix++
	}
	return prefix
}

func (e *HybridCacheEntry) Free() {
	if e == nil {
		return
	}
	for _, c := range e.Caches {
		if c != nil {
			c.Free()
		}
	}
}

func (e *HybridCacheEntry) cachesSlice() []cache.Cache {
	if e == nil {
		return nil
	}
	return e.Caches
}

func (e *HybridCacheEntry) cachesCanTrim() bool {
	if e == nil {
		return false
	}
	for _, c := range e.Caches {
		if c == nil {
			continue
		}
		if !c.CanTrim() {
			return false
		}
	}
	return true
}

func (e *HybridCacheEntry) TrimToPrefix(prefix int) {
	if e == nil {
		return
	}
	for _, c := range e.Caches {
		if c == nil || !c.CanTrim() {
			continue
		}
		trim := c.Offset() - prefix
		if trim > 0 {
			c.Trim(trim)
		}
	}
	if prefix < len(e.Tokens) {
		e.Tokens = e.Tokens[:prefix]
	}
}

func (e *HybridCacheEntry) RestoreToPrefix(target int) (int, bool) {
	if e == nil {
		return 0, false
	}
	restorePos := -1
	sawNonTrimmable := false

	for _, c := range e.Caches {
		if c == nil || c.CanTrim() {
			continue
		}
		sawNonTrimmable = true

		restorer, ok := c.(cache.CheckpointRestorer)
		if !ok {
			return 0, false
		}
		pos, ok := restorer.BestCheckpoint(target)
		if !ok {
			return 0, false
		}
		if restorePos < 0 {
			restorePos = pos
			continue
		}
		if pos != restorePos {
			return 0, false
		}
	}

	if !sawNonTrimmable || restorePos < 0 {
		return 0, false
	}

	e.TrimToPrefix(restorePos)
	for _, c := range e.Caches {
		if c == nil || c.CanTrim() {
			continue
		}
		restorer, ok := c.(cache.CheckpointRestorer)
		if !ok || !restorer.RestoreCheckpoint(restorePos) {
			return 0, false
		}
	}

	if restorePos < len(e.Tokens) {
		e.Tokens = e.Tokens[:restorePos]
	}
	return restorePos, true
}

// FindNearestCache finds the longest common prefix between tokens and the cached sequence
func (r *Runner) FindNearestCache(tokens []int32) ([]cache.Cache, []int32) {
	entries := r.cacheStore()
	if len(entries) == 0 {
		slog.Info("Cache miss", "left", len(tokens))
		return nil, tokens
	}

	branchLimit := promptCacheBranchLimit()
	idx, prefix := r.bestCacheEntry(tokens)
	if idx < 0 {
		if branchLimit <= 1 && len(entries) == 1 && entries[0] != nil {
			entries[0].Free()
			r.setCacheStore(nil)
		}
		slog.Info("Cache miss", "left", len(tokens))
		return nil, tokens
	}
	if idx > 0 {
		r.touchCacheEntry(idx)
	}
	base := r.cache
	if base == nil {
		slog.Info("Cache miss", "left", len(tokens))
		return nil, tokens
	}

	working := base
	forked := false
	if branchLimit > 1 && prefix > 0 {
		working = base.Clone()
		forked = true
	}

	switch {
	case prefix == 0:
		if !forked && branchLimit <= 1 {
			base.Free()
			r.setCacheStore(nil)
		}
		slog.Info("Cache miss", "left", len(tokens))
		return nil, tokens
	case prefix < len(working.Tokens):
		if !working.cachesCanTrim() {
			if restorePos, ok := working.RestoreToPrefix(prefix); ok {
				slog.Info("Cache restore", "total", len(tokens), "matched", prefix, "restored", restorePos, "left", len(tokens[restorePos:]))
				return working.cachesSlice(), tokens[restorePos:]
			}

			if forked {
				working.Free()
			} else if branchLimit <= 1 {
				base.Free()
				r.setCacheStore(nil)
			}
			slog.Info("Cache miss", "left", len(tokens), "matched", prefix, "reason", "non_trimmable_divergence")
			return nil, tokens
		}
		working.TrimToPrefix(prefix)
	}

	slog.Info("Cache hit", "total", len(tokens), "cached", prefix, "left", len(tokens[prefix:]))
	return working.cachesSlice(), tokens[prefix:]
}

func (r *Runner) InsertCache(tokens []int32, caches []cache.Cache) {
	entry := &HybridCacheEntry{
		Tokens: cloneTokens(tokens),
		Caches: caches,
	}

	branchLimit := promptCacheBranchLimit()
	if branchLimit <= 1 {
		r.setCacheStore([]*HybridCacheEntry{entry})
		return
	}

	entries := r.cacheStore()
	// Replace any exact-token duplicate branch with the new result.
	for i := 0; i < len(entries); i++ {
		if entries[i] == nil || !equalTokens(entries[i].Tokens, entry.Tokens) {
			continue
		}
		entries[i].Free()
		entries = append(entries[:i], entries[i+1:]...)
		break
	}

	entries = append([]*HybridCacheEntry{entry}, entries...)
	if len(entries) > branchLimit {
		for _, evicted := range entries[branchLimit:] {
			if evicted != nil {
				evicted.Free()
			}
		}
		entries = entries[:branchLimit]
	}
	r.setCacheStore(entries)
}

func (c *HybridCacheEntry) Clone() *HybridCacheEntry {
	if c == nil {
		return nil
	}
	tokens := make([]int32, len(c.Tokens))
	copy(tokens, c.Tokens)
	caches := make([]cache.Cache, len(c.Caches))
	for i, cc := range c.Caches {
		if cc != nil {
			caches[i] = cc.Clone()
		}
	}
	return &HybridCacheEntry{
		Tokens: tokens,
		Caches: caches,
	}
}

func (c *HybridCacheEntry) LogCache() {
	if c == nil || len(c.Caches) == 0 {
		return
	}
	var totalBytes int
	for _, kv := range c.Caches {
		if kv == nil {
			continue
		}
		k, v := kv.State()
		if k == nil || v == nil {
			continue
		}
		totalBytes += k.NumBytes() + v.NumBytes()
	}
	logutil.Trace(fmt.Sprintf("kv cache tokens: %d, size: %s", c.Caches[0].Offset(), mlx.PrettyBytes(totalBytes)))
}
