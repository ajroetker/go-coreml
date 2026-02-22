# Plan: Decompose High-Rank Reshapes for CoreML

## Context

Florence-2's DaViT vision encoder uses window attention that creates rank-6 intermediate tensors via reshape:
`[batch, h, w, c]` → `[batch, h_win, win_h, w_win, win_w, c]` (rank 6).
CoreML's runtime limits reshape to rank ≤ 5, causing compilation failure:
```
Rank of the shape parameter must be between 0 and 5 (inclusive) in reshape
```

There are 24 such reshapes in the model. The standard pattern is always
`reshape(rank≥6) → transpose → reshape(rank≤5)` — the high-rank intermediate
is immediately transposed and collapsed. Apple's coremltools has a built-in graph
pass `expand_high_rank_reshape_and_transpose` that decomposes these into rank-4
operations. We need to port this pass to go-coreml's `model.Builder.Build()`.

## Algorithm (from Apple's coremltools)

### Pattern Match
Find three consecutive operations: `reshape₁ → transpose → reshape₂` where:
- `reshape₁` output rank ≥ 6
- `reshape₂` output rank ≤ 5
- Intermediate values each have exactly one consumer
- Intermediate values are not model outputs

### Decomposition
1. **Group consecutive axes** in the transpose permutation. E.g., perm `[0,1,3,4,2,5]` groups as `[[0,1],[3,4],[2],[5]]`
2. **Compute merged shape**: product of original shape dims within each group
3. **Compute merged perm**: permutation of the groups
4. If merged rank < 6: emit one reshape (to merged shape) + one transpose (merged perm)
5. If merged rank ≥ 6: emit iterative rank-4 `reshape + transpose([0,2,1,3])` pairs that bubble each axis into place
6. Final reshape to the original `reshape₂` output shape, **reusing the original output name** so downstream references remain valid

### `_get_prod` Helper
Product of shape elements from `start` to `end`, skipping indices in a memo set:
```
_get_prod(start, end, shape, skip) = ∏ shape[i] for i in [start,end) if i not in skip
```

### Iterative Rank-4 Decomposition (rank ≥ 6 path)
```
leading_dim = 1
memo = {}
for i in range(rank):
    axis = perm[i]
    dim = shape[axis]
    memo.add(axis)
    reshape_shape = [leading_dim, _get_prod(0, axis, shape, memo), dim, _get_prod(axis+1, rank, shape, memo)]
    x = reshape(x, reshape_shape)
    x = transpose(x, [0, 2, 1, 3])
    leading_dim *= dim
```

## Changes

### 1. Create `go-coreml/model/optimize.go`

New file containing the graph pass with these functions:

**`(b *Builder) expandHighRankReshapes()`** — Main entry point called from `Build()`:
- Build a consumer map: `map[string][]int` mapping value name → indices of consuming operations
- Scan operations for "reshape" ops with output rank ≥ 6
- For each match, check if the single consumer is "transpose", and its single consumer is "reshape" with rank ≤ 5
- Check intermediates are not in `b.outputs`
- Call decomposition function, collect replacement operations
- Rebuild `b.operations` slice, replacing each matched triple with its replacement sequence
- Update `b.values` map for new intermediate values

**`decomposeReshapeTranspose(...) []*milspec.Operation`** — Decomposition logic:
- Extract: input name, high-rank shape, perm, final shape, output name, dtype
- Group consecutive axes in perm
- Compute merged shape and merged perm
- If merged rank < 6: emit reshape + transpose
- If merged rank ≥ 6: emit iterative rank-4 pairs
- Emit final reshape with the **original output name** of reshape₂

**Helper functions:**
- `getOpOutputShape(op) []int64` — extract shape from operation's output TensorType dimensions
- `getOpOutputDType(op) DType` — extract dtype from operation's output TensorType
- `getOpInputName(op, paramName) string` — extract the name reference from an input argument
- `getInlineInt32s(op, paramName) []int32` — extract inline Int32 constant values from an argument
- `makeReshapeOp(inputName, outputName string, shape []int64, dtype DType) *milspec.Operation` — create a reshape operation in protobuf
- `makeTransposeOp(inputName, outputName string, perm []int64, inputShape []int64, dtype DType) *milspec.Operation` — create a transpose operation in protobuf
- `getProd(start, end int, shape []int64, skip map[int]bool) int64` — product of shape elements

### 2. Modify `go-coreml/model/builder.go` — Call the pass in `Build()`

Add `b.expandHighRankReshapes()` at the start of `Build()` (line ~430), before the operations are packaged into the Block:
```go
func (b *Builder) Build() *Program {
    // Optimize: decompose high-rank reshape patterns for CoreML compatibility.
    b.expandHighRankReshapes()

    // Build function inputs
    inputs := make([]*milspec.NamedValueType, len(b.inputs))
    ...
```

### 3. Create `go-coreml/model/optimize_test.go`

Unit tests for the graph pass:
- **TestExpandRank6ReshapeTranspose**: Build a graph with `reshape([2,3] → [1,1,2,1,3,1]) → transpose → reshape(...)`, verify the resulting operations are all rank ≤ 5
- **TestExpandRank5Passthrough**: Build a graph with rank-5 reshape, verify no transformation occurs
- **TestExpandWindowAttentionPattern**: Simulate DaViT's exact pattern: `[1,14,14,192] → reshape [1,2,7,2,7,192] → transpose [0,1,3,2,4,5] → reshape [1,4,7,7,192]`

## Key Implementation Details

### Protobuf Access Patterns

Extract input name:
```go
op.Inputs["x"].Arguments[0].GetName()
```

Extract inline Int32 constant:
```go
op.Inputs["perm"].Arguments[0].GetValue().GetImmediateValue().GetTensor().GetInts().Values
```

Extract output shape:
```go
for _, dim := range op.Outputs[0].Type.GetTensorType().Dimensions {
    size := int64(dim.GetConstant().Size)
}
```

### Name Preservation

The final reshape in the replacement sequence **must** use the same output name as the original `reshape₂`. This ensures all downstream operations that reference that name continue to work. Intermediate operations use `b.genName()` for unique names.

### Existing Utilities to Reuse

- `toInt32Slice([]int64) []int32` — `model/ops.go:1763`
- `createValue(dtype, shape, data) *milspec.Value` — `model/builder.go:186`
- `b.genName(prefix) string` — `model/builder.go:113`

## Files Modified

| File | Change |
|------|--------|
| `go-coreml/model/optimize.go` | **New** — graph pass implementation |
| `go-coreml/model/optimize_test.go` | **New** — unit tests |
| `go-coreml/model/builder.go` | Add `b.expandHighRankReshapes()` call in `Build()` |

## Verification

1. Run go-coreml model package tests:
   ```bash
   cd /Users/ajroetker/go/src/github.com/gomlx/go-coreml && go test ./model/ -v -count=1
   ```

2. Run full go-coreml test suite:
   ```bash
   cd /Users/ajroetker/go/src/github.com/gomlx/go-coreml && go test ./... -v -count=1
   ```

3. Run Florence-2 CoreML E2E test:
   ```bash
   cd /Users/ajroetker/go/src/github.com/antflydb/antfly/termite && GOEXPERIMENT=simd go test -v -tags coreml ./e2e/ -run TestFlorence2CoreMLSingleStep -timeout 10m -count=1
   ```
