//go:build mlx

package cache

import (
	"os"
	"strconv"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const (
	defaultRecurrentCheckpointCount    = 32
	defaultRecurrentCheckpointInterval = 128
	defaultRecurrentCheckpointMinPos   = 16
)

type recurrentCheckpoint struct {
	pos        int
	convState  *mlx.Array
	deltaState *mlx.Array
}

type recurrentSlot struct {
	convState  *mlx.Array
	deltaState *mlx.Array

	checkpoints       []recurrentCheckpoint
	checkpointSize    int
	checkpointNext    int
	checkpointLastPos int

	refs int
}

func getenvInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func recurrentCheckpointConfig() (count, interval, minPos int) {
	count = getenvInt("OLLAMA_MLX_RECURRENT_CHECKPOINTS", defaultRecurrentCheckpointCount)
	interval = getenvInt("OLLAMA_MLX_RECURRENT_CHECKPOINT_INTERVAL", defaultRecurrentCheckpointInterval)
	minPos = getenvInt("OLLAMA_MLX_RECURRENT_CHECKPOINT_MIN_POS", defaultRecurrentCheckpointMinPos)

	if count < 0 {
		count = 0
	}
	if interval < 0 {
		interval = 0
	}
	if minPos < 0 {
		minPos = 0
	}
	return count, interval, minPos
}

// RecurrentCache stores state for linear-recurrent layers.
//
// Conv state shape: [B, convTail, convDim]
// Delta state shape: [B, numVHeads, headVDim, headKDim]
type RecurrentCache struct {
	slot   *recurrentSlot
	offset int

	convTail  int
	convDim   int
	numVHeads int
	headVDim  int
	headKDim  int

	checkpointCount    int
	checkpointInterval int
	checkpointMinPos   int
}

func newRecurrentSlot(checkpointCount int) *recurrentSlot {
	s := &recurrentSlot{
		refs:              1,
		checkpointLastPos: -1,
	}
	if checkpointCount > 0 {
		s.checkpoints = make([]recurrentCheckpoint, checkpointCount)
		for i := range s.checkpoints {
			s.checkpoints[i].pos = -1
		}
	}
	return s
}

func retainRecurrentSlot(s *recurrentSlot) *recurrentSlot {
	if s != nil {
		s.refs++
	}
	return s
}

func (c *RecurrentCache) setStateMaterialized(dst **mlx.Array, v *mlx.Array) {
	if v == nil || !v.Valid() {
		return
	}
	if *dst == v {
		return
	}

	// Break dependency chains so recurrent state does not retain the full
	// per-token compute graph over time.
	snap := mlx.Snapshot(v)
	mlx.Eval(snap)

	old := *dst
	*dst = snap
	mlx.Pin(snap)

	// Release previous cached state root, then recursively free the transient
	// incoming graph root now that a detached snapshot is retained in cache.
	if old != nil && old != snap {
		mlx.Release(old)
	}
	if v != snap && v != old {
		mlx.Release(v)
	}
}

func (c *RecurrentCache) setStateRaw(dst **mlx.Array, v *mlx.Array) {
	if v == nil || !v.Valid() {
		return
	}
	old := *dst
	*dst = v
	mlx.Pin(v)
	if old != nil && old != v {
		mlx.Release(old)
	}
}

func (c *RecurrentCache) setStateDetached(dst **mlx.Array, v *mlx.Array, ensureContiguous bool) {
	if v == nil || !v.Valid() {
		return
	}
	if *dst == v {
		return
	}

	root := v
	if ensureContiguous {
		root = mlx.Contiguous(v, false)
	}
	detached := mlx.Detach(root)

	old := *dst
	*dst = detached
	mlx.Pin(detached)
	if old != nil && old != detached {
		mlx.Release(old)
	}

	// Intentionally do not force-release root/v here. In the fast path, the detached
	// handle aliases the same MLX value and may still be lazily computed. Releasing the
	// source handles can invalidate the cached state before the next eval/sweep point.
}

func snapshotPinned(a *mlx.Array) *mlx.Array {
	if a == nil || !a.Valid() {
		return nil
	}
	snap := mlx.Snapshot(a)
	mlx.Eval(snap)
	mlx.Pin(snap)
	return snap
}

func NewRecurrentCache(convTail, convDim, numVHeads, headVDim, headKDim int32) *RecurrentCache {
	count, interval, minPos := recurrentCheckpointConfig()
	c := &RecurrentCache{
		slot:               newRecurrentSlot(count),
		convTail:           int(convTail),
		convDim:            int(convDim),
		numVHeads:          int(numVHeads),
		headVDim:           int(headVDim),
		headKDim:           int(headKDim),
		checkpointCount:    count,
		checkpointInterval: interval,
		checkpointMinPos:   minPos,
	}
	return c
}

func clonePinned(a *mlx.Array) *mlx.Array {
	if a == nil || !a.Valid() {
		return nil
	}
	clone := a.Clone()
	mlx.Pin(clone)
	return clone
}

func releaseCheckpointEntry(e *recurrentCheckpoint) {
	mlx.Release(e.convState, e.deltaState)
	e.convState, e.deltaState = nil, nil
	e.pos = -1
}

func releaseRecurrentSlot(s *recurrentSlot) {
	if s == nil {
		return
	}
	s.refs--
	if s.refs > 0 {
		return
	}
	mlx.Release(s.convState, s.deltaState)
	s.convState, s.deltaState = nil, nil
	for i := range s.checkpoints {
		releaseCheckpointEntry(&s.checkpoints[i])
	}
	s.checkpointSize = 0
	s.checkpointNext = 0
	s.checkpointLastPos = -1
}

func cloneRecurrentSlot(src *recurrentSlot) *recurrentSlot {
	if src == nil {
		return nil
	}
	dst := &recurrentSlot{
		checkpointSize:    src.checkpointSize,
		checkpointNext:    src.checkpointNext,
		checkpointLastPos: src.checkpointLastPos,
		refs:              1,
	}
	if src.convState != nil && src.convState.Valid() {
		dst.convState = snapshotPinned(src.convState)
	}
	if src.deltaState != nil && src.deltaState.Valid() {
		dst.deltaState = snapshotPinned(src.deltaState)
	}
	if len(src.checkpoints) > 0 {
		dst.checkpoints = make([]recurrentCheckpoint, len(src.checkpoints))
		for i := range src.checkpoints {
			dst.checkpoints[i].pos = src.checkpoints[i].pos
			if src.checkpoints[i].pos < 0 {
				continue
			}
			dst.checkpoints[i].convState = snapshotPinned(src.checkpoints[i].convState)
			dst.checkpoints[i].deltaState = snapshotPinned(src.checkpoints[i].deltaState)
		}
	}
	return dst
}

func (c *RecurrentCache) slotOrInit() *recurrentSlot {
	if c.slot == nil {
		c.slot = newRecurrentSlot(c.checkpointCount)
	}
	return c.slot
}

func (c *RecurrentCache) ensureWritableSlot() *recurrentSlot {
	s := c.slotOrInit()
	if s.refs <= 1 {
		return s
	}
	c.slot = cloneRecurrentSlot(s)
	s.refs--
	return c.slot
}

func (c *RecurrentCache) pruneCheckpointsAfter(pos int) {
	s := c.ensureWritableSlot()
	if len(s.checkpoints) == 0 {
		return
	}

	size := 0
	next := -1
	last := -1
	minPos := int(^uint(0) >> 1)
	minIdx := 0
	for i := range s.checkpoints {
		e := &s.checkpoints[i]
		if e.pos > pos {
			releaseCheckpointEntry(e)
		}
		if e.pos >= 0 {
			size++
			if e.pos > last {
				last = e.pos
			}
			if e.pos < minPos {
				minPos = e.pos
				minIdx = i
			}
		} else if next == -1 {
			next = i
		}
	}

	s.checkpointSize = size
	s.checkpointLastPos = last
	if size == 0 {
		s.checkpointNext = 0
		return
	}
	if next != -1 {
		s.checkpointNext = next
		return
	}
	s.checkpointNext = minIdx
}

func (c *RecurrentCache) ensure(batch int, dtype mlx.DType) {
	if batch <= 0 {
		batch = 1
	}

	s := c.slotOrInit()
	needConv := s.convState == nil || s.convState.DType() != dtype ||
		s.convState.Dim(0) != batch || s.convState.Dim(1) != c.convTail || s.convState.Dim(2) != c.convDim
	needDelta := s.deltaState == nil || s.deltaState.DType() != dtype ||
		s.deltaState.Dim(0) != batch || s.deltaState.Dim(1) != c.numVHeads || s.deltaState.Dim(2) != c.headVDim || s.deltaState.Dim(3) != c.headKDim
	if !needConv && !needDelta {
		return
	}

	s = c.ensureWritableSlot()
	if needConv {
		c.setStateRaw(&s.convState, mlx.Zeros(dtype, batch, c.convTail, c.convDim))
	}

	if needDelta {
		c.setStateRaw(&s.deltaState, mlx.Zeros(dtype, batch, c.numVHeads, c.headVDim, c.headKDim))
	}
}

func (c *RecurrentCache) ConvState(batch int, dtype mlx.DType) *mlx.Array {
	c.ensure(batch, dtype)
	return c.slotOrInit().convState
}

func (c *RecurrentCache) SetConvState(v *mlx.Array) {
	s := c.ensureWritableSlot()
	c.setStateMaterialized(&s.convState, v)
}

// SetConvStateFast stores conv state without forcing an immediate snapshot/eval.
// Use only for decode hot paths that accept higher transient memory until the next
// sync/sweep point. The conv-state input is usually a slice view, so request a
// compact contiguous copy to avoid pinning the whole source buffer.
func (c *RecurrentCache) SetConvStateFast(v *mlx.Array) {
	s := c.ensureWritableSlot()
	c.setStateDetached(&s.convState, v, true)
}

func (c *RecurrentCache) DeltaState(batch int, dtype mlx.DType) *mlx.Array {
	c.ensure(batch, dtype)
	return c.slotOrInit().deltaState
}

func (c *RecurrentCache) SetDeltaState(v *mlx.Array) {
	s := c.ensureWritableSlot()
	c.setStateMaterialized(&s.deltaState, v)
}

// SetDeltaStateFast stores delta state without forcing an immediate snapshot/eval.
// Use only for decode hot paths that accept higher transient memory until the next
// sync/sweep point.
func (c *RecurrentCache) SetDeltaStateFast(v *mlx.Array) {
	s := c.ensureWritableSlot()
	c.setStateDetached(&s.deltaState, v, false)
}

func (c *RecurrentCache) Advance(n int) {
	c.offset += n
}

func (c *RecurrentCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	return keys, values
}

func (c *RecurrentCache) State() (*mlx.Array, *mlx.Array) {
	c.ensure(1, mlx.DTypeFloat32)
	s := c.slotOrInit()
	return s.convState, s.deltaState
}

func (c *RecurrentCache) Materialize() []*mlx.Array {
	out := make([]*mlx.Array, 0, 2)
	s := c.slot
	if s == nil {
		return out
	}
	if s.convState != nil && s.convState.Valid() {
		out = append(out, s.convState)
	}
	if s.deltaState != nil && s.deltaState.Valid() {
		out = append(out, s.deltaState)
	}
	return out
}

func (c *RecurrentCache) RecordCheckpoint(pos int) {
	s := c.slot
	if s == nil || len(s.checkpoints) == 0 || pos <= 0 || pos < c.checkpointMinPos {
		return
	}
	if c.offset != pos {
		// Checkpoints are keyed by logical token position. Ignore callers with a
		// mismatched position to avoid restoring inconsistent recurrent state.
		return
	}
	if s.convState == nil || s.deltaState == nil || !s.convState.Valid() || !s.deltaState.Valid() {
		return
	}
	if s.checkpointLastPos == pos {
		return
	}
	if s.checkpointLastPos >= 0 && c.checkpointInterval > 0 && pos-s.checkpointLastPos < c.checkpointInterval {
		return
	}
	if s.refs > 1 {
		s = c.ensureWritableSlot()
	}

	idx := s.checkpointNext
	e := &s.checkpoints[idx]
	releaseCheckpointEntry(e)
	e.pos = pos
	e.convState = clonePinned(s.convState)
	e.deltaState = clonePinned(s.deltaState)

	s.checkpointNext = (idx + 1) % len(s.checkpoints)
	if s.checkpointSize < len(s.checkpoints) {
		s.checkpointSize++
	}
	s.checkpointLastPos = pos
}

func (c *RecurrentCache) BestCheckpoint(target int) (pos int, ok bool) {
	s := c.slot
	if s == nil {
		return 0, false
	}
	best := -1
	for i := range s.checkpoints {
		pos := s.checkpoints[i].pos
		if pos < 0 || pos > target {
			continue
		}
		if pos > best {
			best = pos
		}
	}
	if best < 0 {
		return 0, false
	}
	return best, true
}

func (c *RecurrentCache) RestoreCheckpoint(pos int) bool {
	if pos < 0 {
		return false
	}
	s := c.ensureWritableSlot()
	for i := range s.checkpoints {
		e := &s.checkpoints[i]
		if e.pos != pos {
			continue
		}
		if e.convState == nil || e.deltaState == nil || !e.convState.Valid() || !e.deltaState.Valid() {
			return false
		}

		c.setStateRaw(&s.convState, e.convState.Clone())
		c.setStateRaw(&s.deltaState, e.deltaState.Clone())
		c.offset = pos
		c.pruneCheckpointsAfter(pos)
		return true
	}
	return false
}

func (c *RecurrentCache) CanTrim() bool { return false }

func (c *RecurrentCache) Trim(n int) int {
	// Recurrent state is not directly trimmable; callers should use
	// checkpoint-based restore instead.
	_ = n
	return 0
}

func (c *RecurrentCache) Clone() Cache {
	clone := &RecurrentCache{
		slot:               retainRecurrentSlot(c.slotOrInit()),
		offset:             c.offset,
		convTail:           c.convTail,
		convDim:            c.convDim,
		numVHeads:          c.numVHeads,
		headVDim:           c.headVDim,
		headKDim:           c.headKDim,
		checkpointCount:    c.checkpointCount,
		checkpointInterval: c.checkpointInterval,
		checkpointMinPos:   c.checkpointMinPos,
	}
	return clone
}

func (c *RecurrentCache) Free() {
	releaseRecurrentSlot(c.slot)
	c.slot = nil
	c.offset = 0
}

func (c *RecurrentCache) Offset() int { return c.offset }
func (c *RecurrentCache) Len() int    { return c.offset }
