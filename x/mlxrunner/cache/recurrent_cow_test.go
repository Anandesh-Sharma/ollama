//go:build mlx

package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func requireMLXRuntime(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX runtime unavailable: %v", err)
	}
}

func newTestRecurrentCache(t *testing.T) *RecurrentCache {
	t.Helper()
	requireMLXRuntime(t)
	t.Setenv("OLLAMA_MLX_RECURRENT_CHECKPOINTS", "2")
	t.Setenv("OLLAMA_MLX_RECURRENT_CHECKPOINT_INTERVAL", "0")
	t.Setenv("OLLAMA_MLX_RECURRENT_CHECKPOINT_MIN_POS", "0")

	c := NewRecurrentCache(2, 3, 1, 2, 2)
	_ = c.ConvState(1, mlx.DTypeFloat32)
	_ = c.DeltaState(1, mlx.DTypeFloat32)
	return c
}

func TestRecurrentCacheCloneSharesSlotUntilMutation(t *testing.T) {
	c1 := newTestRecurrentCache(t)
	t.Cleanup(func() {
		c1.Free()
		mlx.Sweep()
	})

	c1.Advance(8)
	c1.RecordCheckpoint(8)
	if got, ok := c1.BestCheckpoint(8); !ok || got != 8 {
		t.Fatalf("BestCheckpoint(8) = (%d, %v), want (8, true)", got, ok)
	}

	c2 := c1.Clone().(*RecurrentCache)
	t.Cleanup(func() {
		c2.Free()
		mlx.Sweep()
	})

	if c1.slot == nil || c2.slot == nil {
		t.Fatal("expected non-nil shared slots")
	}
	if c1.slot != c2.slot {
		t.Fatal("clone did not share recurrent slot")
	}
	if c1.slot.refs != 2 {
		t.Fatalf("shared slot refs = %d, want 2", c1.slot.refs)
	}

	// Read access should not trigger a COW detach.
	_ = c2.ConvState(1, mlx.DTypeFloat32)
	_ = c2.DeltaState(1, mlx.DTypeFloat32)
	if c1.slot != c2.slot {
		t.Fatal("read access detached shared recurrent slot")
	}

	// Mutating recurrent state should detach and deep-copy checkpoint metadata.
	c2.SetConvState(mlx.Zeros(mlx.DTypeFloat32, 1, 2, 3))

	if c1.slot == c2.slot {
		t.Fatal("SetConvState did not detach shared recurrent slot")
	}
	if c1.slot.refs != 1 || c2.slot.refs != 1 {
		t.Fatalf("post-detach refs = (%d, %d), want (1, 1)", c1.slot.refs, c2.slot.refs)
	}
	if len(c1.slot.checkpoints) == 0 || len(c2.slot.checkpoints) == 0 {
		t.Fatal("expected checkpoint ring to be preserved on detach")
	}
	if c1.slot.checkpoints[0].pos != c2.slot.checkpoints[0].pos {
		t.Fatalf("checkpoint pos mismatch after detach: %d vs %d", c1.slot.checkpoints[0].pos, c2.slot.checkpoints[0].pos)
	}
	if c1.slot.checkpoints[0].pos != 8 {
		t.Fatalf("checkpoint pos = %d, want 8", c1.slot.checkpoints[0].pos)
	}
	if c1.slot.checkpoints[0].convState == c2.slot.checkpoints[0].convState {
		t.Fatal("checkpoint conv state was aliased after COW detach")
	}
	if c1.slot.checkpoints[0].deltaState == c2.slot.checkpoints[0].deltaState {
		t.Fatal("checkpoint delta state was aliased after COW detach")
	}
}

func TestRecurrentCacheFreeKeepsSharedCloneAlive(t *testing.T) {
	c1 := newTestRecurrentCache(t)
	c2 := c1.Clone().(*RecurrentCache)
	t.Cleanup(func() {
		c1.Free()
		c2.Free()
		mlx.Sweep()
	})

	if c2.slot == nil || c2.slot.refs != 2 {
		t.Fatalf("shared clone refs = %d, want 2", func() int {
			if c2.slot == nil {
				return 0
			}
			return c2.slot.refs
		}())
	}

	c1.Free()

	if c2.slot == nil {
		t.Fatal("clone slot was cleared after freeing sibling clone")
	}
	if c2.slot.refs != 1 {
		t.Fatalf("clone slot refs after sibling Free = %d, want 1", c2.slot.refs)
	}
	if state := c2.ConvState(1, mlx.DTypeFloat32); state == nil || !state.Valid() {
		t.Fatal("clone conv state invalid after freeing sibling clone")
	}
	if state := c2.DeltaState(1, mlx.DTypeFloat32); state == nil || !state.Valid() {
		t.Fatal("clone delta state invalid after freeing sibling clone")
	}
}
