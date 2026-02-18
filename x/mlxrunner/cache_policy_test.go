//go:build mlx

package mlxrunner

import (
	"reflect"
	"testing"

	cachepkg "github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type fakeCache struct {
	canTrim  bool
	trims    []int
	freeCall int
	offset   int
}

func (f *fakeCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) { return keys, values }
func (f *fakeCache) State() (*mlx.Array, *mlx.Array)                         { return nil, nil }
func (f *fakeCache) Materialize() []*mlx.Array                               { return nil }
func (f *fakeCache) CanTrim() bool                                           { return f.canTrim }
func (f *fakeCache) Trim(n int) int {
	f.trims = append(f.trims, n)
	f.offset -= n
	return n
}
func (f *fakeCache) Clone() cachepkg.Cache { return &fakeCache{canTrim: f.canTrim, offset: f.offset} }
func (f *fakeCache) Free()                 { f.freeCall++ }
func (f *fakeCache) Offset() int           { return f.offset }
func (f *fakeCache) Len() int              { return f.offset }

type fakeCheckpointCache struct {
	fakeCache
	bestPos        int
	hasCheckpoint  bool
	restoreCalls   []int
	restoreSuccess bool
}

func (f *fakeCheckpointCache) BestCheckpoint(target int) (int, bool) {
	if !f.hasCheckpoint || f.bestPos > target {
		return 0, false
	}
	return f.bestPos, true
}

func (f *fakeCheckpointCache) RestoreCheckpoint(pos int) bool {
	f.restoreCalls = append(f.restoreCalls, pos)
	if !f.restoreSuccess || pos != f.bestPos {
		return false
	}
	f.offset = pos
	return true
}

func (f *fakeCheckpointCache) Clone() cachepkg.Cache {
	clone := *f
	clone.trims = nil
	clone.restoreCalls = nil
	return &clone
}

func TestFindNearestCacheReusesAppendOnlyNonTrimmableCache(t *testing.T) {
	fc := &fakeCache{canTrim: false, offset: 2}
	r := &Runner{
		cache: &CacheEntry{
			Tokens: []int32{1, 2},
			Caches: []cachepkg.Cache{fc},
		},
	}

	gotCaches, gotTokens := r.FindNearestCache([]int32{1, 2, 3, 4})

	if len(gotCaches) != 1 || gotCaches[0] != fc {
		t.Fatalf("returned caches = %#v, want original cache", gotCaches)
	}
	if want := []int32{3, 4}; !reflect.DeepEqual(gotTokens, want) {
		t.Fatalf("tokens left = %v, want %v", gotTokens, want)
	}
	if fc.freeCall != 0 {
		t.Fatalf("free calls = %d, want 0", fc.freeCall)
	}
	if len(fc.trims) != 0 {
		t.Fatalf("trim calls = %v, want none", fc.trims)
	}
}

func TestFindNearestCacheDropsNonTrimmableCacheOnDivergence(t *testing.T) {
	fc := &fakeCache{canTrim: false, offset: 4}
	r := &Runner{
		cache: &CacheEntry{
			Tokens: []int32{1, 2, 3, 4},
			Caches: []cachepkg.Cache{fc},
		},
	}

	gotCaches, gotTokens := r.FindNearestCache([]int32{1, 2, 9})

	if gotCaches != nil {
		t.Fatalf("returned caches = %#v, want nil", gotCaches)
	}
	if want := []int32{1, 2, 9}; !reflect.DeepEqual(gotTokens, want) {
		t.Fatalf("tokens left = %v, want %v", gotTokens, want)
	}
	if fc.freeCall != 1 {
		t.Fatalf("free calls = %d, want 1", fc.freeCall)
	}
	if len(fc.trims) != 0 {
		t.Fatalf("trim calls = %v, want none", fc.trims)
	}
	if r.cache != nil {
		t.Fatal("runner cache should be cleared on non-trimmable divergence")
	}
}

func TestFindNearestCacheTrimsTrimmableCacheOnDivergence(t *testing.T) {
	fc := &fakeCache{canTrim: true, offset: 4}
	r := &Runner{
		cache: &CacheEntry{
			Tokens: []int32{1, 2, 3, 4},
			Caches: []cachepkg.Cache{fc},
		},
	}

	gotCaches, gotTokens := r.FindNearestCache([]int32{1, 2, 9})

	if len(gotCaches) != 1 || gotCaches[0] != fc {
		t.Fatalf("returned caches = %#v, want original cache", gotCaches)
	}
	if want := []int32{9}; !reflect.DeepEqual(gotTokens, want) {
		t.Fatalf("tokens left = %v, want %v", gotTokens, want)
	}
	if fc.freeCall != 0 {
		t.Fatalf("free calls = %d, want 0", fc.freeCall)
	}
	if want := []int{2}; !reflect.DeepEqual(fc.trims, want) {
		t.Fatalf("trim calls = %v, want %v", fc.trims, want)
	}
	if want := []int32{1, 2}; !reflect.DeepEqual(r.cache.Tokens, want) {
		t.Fatalf("cached tokens = %v, want %v", r.cache.Tokens, want)
	}
}

func TestFindNearestCacheRestoresCheckpointForNonTrimmableCaches(t *testing.T) {
	kv := &fakeCache{canTrim: true, offset: 7}
	rc1 := &fakeCheckpointCache{
		fakeCache:      fakeCache{canTrim: false, offset: 7},
		bestPos:        4,
		hasCheckpoint:  true,
		restoreSuccess: true,
	}
	rc2 := &fakeCheckpointCache{
		fakeCache:      fakeCache{canTrim: false, offset: 7},
		bestPos:        4,
		hasCheckpoint:  true,
		restoreSuccess: true,
	}

	r := &Runner{
		cache: &CacheEntry{
			Tokens: []int32{1, 2, 3, 4, 5, 6, 7},
			Caches: []cachepkg.Cache{kv, rc1, rc2},
		},
	}

	gotCaches, gotTokens := r.FindNearestCache([]int32{1, 2, 3, 4, 8})

	if len(gotCaches) != 3 {
		t.Fatalf("returned caches len = %d, want 3", len(gotCaches))
	}
	if want := []int32{8}; !reflect.DeepEqual(gotTokens, want) {
		t.Fatalf("tokens left = %v, want %v", gotTokens, want)
	}
	if want := []int{3}; !reflect.DeepEqual(kv.trims, want) {
		t.Fatalf("kv trim calls = %v, want %v", kv.trims, want)
	}
	if want := []int{4}; !reflect.DeepEqual(rc1.restoreCalls, want) {
		t.Fatalf("rc1 restore calls = %v, want %v", rc1.restoreCalls, want)
	}
	if want := []int{4}; !reflect.DeepEqual(rc2.restoreCalls, want) {
		t.Fatalf("rc2 restore calls = %v, want %v", rc2.restoreCalls, want)
	}
	if want := []int32{1, 2, 3, 4}; !reflect.DeepEqual(r.cache.Tokens, want) {
		t.Fatalf("cached tokens = %v, want %v", r.cache.Tokens, want)
	}
}

func TestFindNearestCacheDropsOnMismatchedCheckpointRestorePoints(t *testing.T) {
	rc1 := &fakeCheckpointCache{
		fakeCache:      fakeCache{canTrim: false, offset: 7},
		bestPos:        4,
		hasCheckpoint:  true,
		restoreSuccess: true,
	}
	rc2 := &fakeCheckpointCache{
		fakeCache:      fakeCache{canTrim: false, offset: 7},
		bestPos:        3,
		hasCheckpoint:  true,
		restoreSuccess: true,
	}

	r := &Runner{
		cache: &CacheEntry{
			Tokens: []int32{1, 2, 3, 4, 5, 6, 7},
			Caches: []cachepkg.Cache{rc1, rc2},
		},
	}

	gotCaches, gotTokens := r.FindNearestCache([]int32{1, 2, 3, 4, 8})

	if gotCaches != nil {
		t.Fatalf("returned caches = %#v, want nil", gotCaches)
	}
	if want := []int32{1, 2, 3, 4, 8}; !reflect.DeepEqual(gotTokens, want) {
		t.Fatalf("tokens left = %v, want %v", gotTokens, want)
	}
	if rc1.freeCall != 1 || rc2.freeCall != 1 {
		t.Fatalf("free calls = (%d,%d), want (1,1)", rc1.freeCall, rc2.freeCall)
	}
	if len(rc1.restoreCalls) != 0 || len(rc2.restoreCalls) != 0 {
		t.Fatalf("restore calls = (%v,%v), want none", rc1.restoreCalls, rc2.restoreCalls)
	}
}

func TestFindNearestCacheSelectsBestPrefixAcrossBranches(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PROMPT_CACHE_BRANCHES", "4")

	short := &fakeCache{canTrim: true, offset: 2}
	long := &fakeCache{canTrim: true, offset: 4}
	shortEntry := &HybridCacheEntry{
		Tokens: []int32{1, 2},
		Caches: []cachepkg.Cache{short},
	}
	longEntry := &HybridCacheEntry{
		Tokens: []int32{1, 2, 3, 4},
		Caches: []cachepkg.Cache{long},
	}

	r := &Runner{
		cache:  shortEntry,
		caches: []*HybridCacheEntry{shortEntry, longEntry},
	}

	gotCaches, gotTokens := r.FindNearestCache([]int32{1, 2, 3, 4, 9})

	if want := []int32{9}; !reflect.DeepEqual(gotTokens, want) {
		t.Fatalf("tokens left = %v, want %v", gotTokens, want)
	}
	if len(gotCaches) != 1 {
		t.Fatalf("returned caches len = %d, want 1", len(gotCaches))
	}
	if gotCaches[0] == long {
		t.Fatal("expected cloned cache in multi-branch mode, got original branch cache")
	}
	if r.cache != longEntry || r.caches[0] != longEntry {
		t.Fatal("best branch was not promoted to front of cache store")
	}
}

func TestFindNearestCacheForksBranchWithCloneWhenMultiBranchEnabled(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PROMPT_CACHE_BRANCHES", "2")

	base := &fakeCache{canTrim: true, offset: 4}
	baseEntry := &HybridCacheEntry{
		Tokens: []int32{1, 2, 3, 4},
		Caches: []cachepkg.Cache{base},
	}
	r := &Runner{
		cache:  baseEntry,
		caches: []*HybridCacheEntry{baseEntry},
	}

	gotCaches, gotTokens := r.FindNearestCache([]int32{1, 2, 9})

	if want := []int32{9}; !reflect.DeepEqual(gotTokens, want) {
		t.Fatalf("tokens left = %v, want %v", gotTokens, want)
	}
	if len(gotCaches) != 1 {
		t.Fatalf("returned caches len = %d, want 1", len(gotCaches))
	}
	clone, ok := gotCaches[0].(*fakeCache)
	if !ok {
		t.Fatalf("returned cache type = %T, want *fakeCache", gotCaches[0])
	}
	if clone == base {
		t.Fatal("expected branch fork to return a cloned cache")
	}
	if len(base.trims) != 0 {
		t.Fatalf("base branch trim calls = %v, want none", base.trims)
	}
	if want := []int{2}; !reflect.DeepEqual(clone.trims, want) {
		t.Fatalf("forked branch trim calls = %v, want %v", clone.trims, want)
	}
	if want := []int32{1, 2, 3, 4}; !reflect.DeepEqual(baseEntry.Tokens, want) {
		t.Fatalf("base entry tokens = %v, want %v", baseEntry.Tokens, want)
	}

	r.InsertCache([]int32{1, 2, 9}, gotCaches)
	if len(r.caches) != 2 {
		t.Fatalf("cache store len = %d, want 2", len(r.caches))
	}
	if want := []int32{1, 2, 9}; !reflect.DeepEqual(r.caches[0].Tokens, want) {
		t.Fatalf("new branch tokens = %v, want %v", r.caches[0].Tokens, want)
	}
	if want := []int32{1, 2, 3, 4}; !reflect.DeepEqual(r.caches[1].Tokens, want) {
		t.Fatalf("preserved branch tokens = %v, want %v", r.caches[1].Tokens, want)
	}
}

func TestInsertCacheEvictsOldestBranchWhenStoreFull(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PROMPT_CACHE_BRANCHES", "2")

	f1 := &fakeCache{canTrim: true, offset: 1}
	f2 := &fakeCache{canTrim: true, offset: 2}
	f3 := &fakeCache{canTrim: true, offset: 3}
	r := &Runner{}

	r.InsertCache([]int32{1}, []cachepkg.Cache{f1})
	r.InsertCache([]int32{1, 2}, []cachepkg.Cache{f2})
	r.InsertCache([]int32{1, 2, 3}, []cachepkg.Cache{f3})

	if len(r.caches) != 2 {
		t.Fatalf("cache store len = %d, want 2", len(r.caches))
	}
	if f1.freeCall != 1 {
		t.Fatalf("oldest branch free calls = %d, want 1", f1.freeCall)
	}
	if f2.freeCall != 0 || f3.freeCall != 0 {
		t.Fatalf("unexpected frees for retained branches: f2=%d f3=%d", f2.freeCall, f3.freeCall)
	}
	if want := []int32{1, 2, 3}; !reflect.DeepEqual(r.caches[0].Tokens, want) {
		t.Fatalf("MRU tokens = %v, want %v", r.caches[0].Tokens, want)
	}
	if want := []int32{1, 2}; !reflect.DeepEqual(r.caches[1].Tokens, want) {
		t.Fatalf("LRU tokens = %v, want %v", r.caches[1].Tokens, want)
	}
}
