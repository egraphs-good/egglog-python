# Issue #396: Duplicate Python Values - Analysis and Solutions

## Problem Summary

When Python objects are stored in egglog using `PyObject`, cloudpickle is used to serialize them to bytes. These bytes are then used for equality comparison and hashing in the Rust layer (`PyPickledValue`).

**The Issue**: Cloudpickle does not always produce identical byte sequences for the same object when pickled by-value. This particularly affects:
- Classes defined in `__main__` module
- Closures and nested functions
- Objects that cannot be imported normally

When cloudpickle produces different bytes for the same logical object, the e-graph treats them as distinct nodes, creating duplicates.

## Architecture Analysis

### Current Flow:
1. Python object → `cloudpickle.dumps()` → bytes
2. Bytes stored in `PyObjectDecl.pickled` (Python) and `PyPickledValue` (Rust)
3. Rust uses byte-level equality: `impl Eq for PyPickledValue { ... }`
4. If bytes differ → different e-graph nodes

### Root Cause:
The `PyPickledValue` struct in `src/py_object_sort.rs` implements equality based on raw bytes:
```rust
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct PyPickledValue(pub Vec<u8>);
```

## Proposed Solutions

### Solution 1: Object Identity Cache ⭐ (RECOMMENDED)
**Approach**: Maintain a Python-side cache that maps object identity to pickled bytes.

**Implementation**:
- Add a cache in the EGraph: `Dict[int, bytes]` mapping `id(obj)` to pickled bytes
- When creating a PyObject:
  1. Check if `id(obj)` exists in cache
  2. If yes, reuse those bytes
  3. If no, pickle and cache
- Use weak references to avoid keeping objects alive unnecessarily

**Pros**:
- ✅ Fast - no repeated pickling
- ✅ Solves the common case where same object is added multiple times
- ✅ Minimal changes to existing architecture
- ✅ Preserves object identity semantics

**Cons**:
- ❌ Only works while objects are alive
- ❌ Doesn't help with value equality (two equal but distinct objects)
- ❌ Requires Python-side state management

**Example Code Location**: `python/egglog/egraph_state.py` - add cache to EGraphState

---

### Solution 2: Hash-Based Deduplication
**Approach**: Use Python's `hash()` for hashable objects, fall back to bytes for others.

**Implementation**:
- Modify `dump()` to compute and store both:
  - Pickled bytes (for reconstruction)
  - Python hash (for equality)
- In Rust, compare using stored hash for hashable objects
- Keep objects alive in a registry

**Pros**:
- ✅ Respects Python's hash semantics
- ✅ Works for hashable objects

**Cons**:
- ❌ Doesn't work for unhashable objects (dicts, lists, etc.)
- ❌ Complex lifecycle management
- ❌ Hash collisions possible

---

### Solution 3: Unpickled Equality Comparison
**Approach**: When comparing PyPickledValue for equality, unpickle and use Python's `==`.

**Implementation**:
- Override `PartialEq` in Rust to unpickle both objects
- Call back to Python to perform equality check
- Cache result to avoid repeated unpickling

**Pros**:
- ✅ Respects Python value equality semantics
- ✅ Conceptually simple

**Cons**:
- ❌ **Very expensive** - unpickle on every equality check
- ❌ May not work for objects without `__eq__`
- ❌ Could be 100-1000x slower
- ❌ Not suitable for use in hash tables

---

### Solution 4: Canonical Pickle Format
**Approach**: Normalize pickled bytes to ensure determinism.

**Implementation**:
- Post-process pickled bytes to remove non-deterministic elements
- Sort dictionaries, normalize timestamps, etc.

**Pros**:
- ✅ Would work with existing architecture

**Cons**:
- ❌ **Very difficult** to implement correctly
- ❌ May break cloudpickle's reconstruction
- ❌ Pickle format is complex and opaque
- ❌ May not be possible for all objects

---

### Solution 5: Hybrid Identity + Value Cache
**Approach**: Combine object identity cache with value-based deduplication.

**Implementation**:
- First check identity cache (fast path)
- For new objects, compute a content-based fingerprint
- Cache both identity and fingerprint mappings

**Pros**:
- ✅ Best of both worlds
- ✅ Handles both identity and value equality

**Cons**:
- ❌ Complex implementation
- ❌ Hard to define "content fingerprint" reliably
- ❌ Still has performance overhead

---

### Solution 6: User-Provided Keys
**Approach**: Allow users to optionally provide a custom key for deduplication.

**Implementation**:
```python
PyObject(obj, key="my_unique_key")
```

**Pros**:
- ✅ Simple to implement
- ✅ User has full control
- ✅ Works for any object

**Cons**:
- ❌ Puts burden on user
- ❌ Not automatic
- ❌ Easy to misuse

---

## Recommendation

**Implement Solution 1 (Object Identity Cache)** as it:
1. Solves the most common case (same object added multiple times)
2. Requires minimal changes
3. Has good performance characteristics
4. Can be implemented entirely in Python layer

### Implementation Plan:

1. **Add cache to EGraph** (`python/egglog/egraph_state.py`):
   ```python
   class EGraphState:
       def __init__(self):
           self._py_object_cache: Dict[int, bytes] = {}
   ```

2. **Modify PyObject creation** to check cache first:
   ```python
   def create_py_object(obj):
       obj_id = id(obj)
       if obj_id in cache:
           return cached_bytes
       pickled = cloudpickle.dumps(obj)
       cache[obj_id] = pickled
       return pickled
   ```

3. **Use weak references** to avoid keeping objects alive:
   ```python
   import weakref
   self._py_object_refs = weakref.WeakValueDictionary()
   ```

4. **Add tests** to verify deduplication works

### Future Enhancements:
- Add optional value-based deduplication for specific types
- Provide user override for custom equality
- Add metrics/warnings when duplicates are detected

## Test Results

Created test file: `python/tests/test_issue_396_duplicate_pyobjects.py`
- Tests demonstrate expected behavior
- In simple cases, cloudpickle IS deterministic
- Real-world cases with classes in `__main__` may still have non-determinism
- Test provides framework to catch regressions

## References

- Issue: https://github.com/egraphs-good/egglog-python/issues/396
- PR Discussion: https://github.com/egraphs-good/egglog-python/pull/393
- Cloudpickle source: https://github.com/cloudpipe/cloudpickle/blob/f5199fe2bc102a5ee070c743336699fc885ca966/cloudpickle/cloudpickle.py#L291-L303
