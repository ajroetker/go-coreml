High Priority Issues

1. Error Handling (1 panic that should be error):
- gomlx/buffer.go:113 - Buffer finalization
- (FIXED) model/ops.go:1325 - Concat with no inputs - now returns nil and sets error
- (FIXED) model/ops.go:1543 - Einsum with wrong input count - now returns nil and sets error

2. Missing Operations for GoMLX backend:
- Call - calling sub-functions
- Sort - sorting with comparator
- While - loop control flow (MIL support implemented in model layer, GoMLX integration pending)
- If - conditional branching (MIL support implemented in model layer, GoMLX integration pending)
- Complex Gather - multi-axis gather

3. Documented TODOs that block use cases:
- MinPool not implemented (could do -MaxPool(-x))
- Input dilation > 1 not supported
- Batch group count > 1 not supported in Conv

Medium Priority

4. Capability declaration mismatches:
- Some ops declared as "supported" but only partially work (Gather, Pad interior, ReduceWindow with min/product)

5. Documentation gaps:
- No limitations document
- No troubleshooting guide
- No performance tuning guide
- (ADDED) Control flow operations documented in docs/control-flow-operations.md

Recent Progress (Control Flow)

The model layer now supports nested blocks for control flow operations:

- `model.BlockBuilder` - Build nested blocks within operations
- `model.Cond` - Conditional execution (if/else) with true/false branch blocks
- `model.WhileLoop` - Loop execution with condition and body blocks
- `serialize_blob.go` - Updated to handle nested blocks in operations

The GoMLX backend still returns NotImplementedError for While/If since full integration
requires closure compilation. Users can:
- Use `Where()` for element-wise conditionals
- Unroll loops at graph construction time
- Use the model layer directly for control flow
