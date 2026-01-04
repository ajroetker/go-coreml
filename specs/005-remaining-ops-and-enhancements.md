# Remaining Operations and Enhancements - Phase 6

## Overview

This document outlines the remaining work to achieve comprehensive CoreML backend coverage for GoMLX. Phases 3-5 implemented 60+ operations. This phase focuses on:

1. Enhanced DotGeneral (batch dimensions, arbitrary axes)
2. Missing gomlx backend wrappers
3. Additional MIL operations
4. Integration testing with real models

## Current Implementation Status

### Implemented in gomlx/backends/coreml (40+ ops)

| Category | Operations |
|----------|-----------|
| Unary Math | Abs, Neg, Exp, Log, Sqrt, Floor, Ceil, Round, Sign, Tanh, Logistic, Cos, Sin, Erf |
| Binary Math | Add, Sub, Mul, Div, Pow, Max, Min |
| Comparison | Equal, NotEqual, LessThan, LessOrEqual, GreaterThan, GreaterOrEqual |
| Shape | Reshape, Transpose, Pad, Reverse |
| Reduction | ReduceSum, ReduceMax, ReduceMin, ReduceProduct, ArgMinMax |
| Matrix | DotGeneral (simple matmul only) |
| Conditional | Where |
| Indexing | Slice, Gather (partial) |
| Convolution | ConvGeneral |
| Normalization | BatchNormForInference |

### Implemented in go-coreml only (not exposed to gomlx)

| Category | Operations |
|----------|-----------|
| Tensor Manipulation | Tile, Concat |
| Convolution | Conv, ConvTranspose, ConvWithBias |
| Pooling | MaxPool, AvgPool, GlobalAvgPool2D, GlobalMaxPool2D |
| Normalization | LayerNorm, InstanceNorm |
| Activations | Gelu, Silu, LeakyRelu, Elu, Softplus, Relu, Sigmoid, Softmax |
| Trig/Hyperbolic | Acos, Asin, Atan, Cosh, Sinh |

---

## Priority 1: Enhanced DotGeneral

The current DotGeneral only supports simple 2D matrix multiplication. Full support requires:

### 1.1 Batch Dimensions

Support batch matmul: `[B, M, K] @ [B, K, N] -> [B, M, N]`

**Implementation Strategy:**

```go
// gomlx/backends/coreml/ops.go
func (b *Builder) DotGeneral(
    lhsOp backends.Op,
    lhsContractingAxes, lhsBatchAxes []int,
    rhsOp backends.Op,
    rhsContractingAxes, rhsBatchAxes []int,
) (backends.Op, error) {
    // Case 1: Simple matmul (existing code)
    // Case 2: Batched matmul - CoreML's matmul handles batch dims natively
    // Case 3: Complex cases - use transpose/reshape to normalize
}
```

**CoreML MatMul Batch Support:**
CoreML's `matmul` already supports batch dimensions:
- `[..., M, K] @ [..., K, N] -> [..., M, N]`
- Batch dimensions must match or broadcast

**Tasks:**
- [ ] Add batch dimension detection in DotGeneral
- [ ] Handle broadcasting of batch dimensions
- [ ] Add tests for batched matmul
- [ ] Test attention mechanism (Q @ K^T @ V)

### 1.2 Arbitrary Contracting Axes

Support contracting on any axis, not just the last axis of LHS.

**Implementation Strategy:**
1. Transpose inputs to move contracting axes to standard positions
2. Call CoreML matmul
3. Transpose output if needed

```go
func normalizeDotGeneralInputs(
    lhs *Node, lhsContractingAxes, lhsBatchAxes []int,
    rhs *Node, rhsContractingAxes, rhsBatchAxes []int,
) (*Node, *Node, []int) {
    // Step 1: Transpose lhs to [batch..., M, K]
    // Step 2: Transpose rhs to [batch..., K, N]
    // Step 3: Return transposed nodes and output permutation
}
```

**Tasks:**
- [ ] Implement axis normalization helper
- [ ] Handle output shape reconstruction
- [ ] Add tests for non-standard contracting axes

---

## Priority 2: Missing GoMLX Backend Wrappers

These operations exist in go-coreml but need gomlx wrappers (where OpTypes exist):

### 2.1 Concatenate (OpTypeConcatenate)

```go
func (b *Builder) Concatenate(operands []backends.Op, axis int) (backends.Op, error) {
    opType := backends.OpTypeConcatenate

    // Validate operands
    nodes := make([]*Node, len(operands))
    milValues := make([]*model.Value, len(operands))
    for i, op := range operands {
        node, err := b.checkOp(opType.String(), op)
        if err != nil {
            return nil, err
        }
        nodes[i] = node
        milValues[i] = node.milValue
    }

    // Use shapeinference.ConcatenateOp
    outputShape, err := shapeinference.ConcatenateOp(axis, shapes...)
    if err != nil {
        return nil, err
    }

    resultValue := b.milBuilder.Concat(milValues, int64(axis))
    node := b.newNodeMultiInput(opType, outputShape, resultValue, nodes)
    return node, nil
}
```

**Tasks:**
- [ ] Add `newNodeMultiInput` helper for multi-input operations
- [ ] Implement Concatenate wrapper
- [ ] Add tests

### 2.2 Broadcast Operations (OpTypeBroadcast, OpTypeBroadcastInDim)

```go
func (b *Builder) Broadcast(operandOp backends.Op, shape shapes.Shape) (backends.Op, error) {
    // Use Tile + ExpandDims to implement broadcast
}

func (b *Builder) BroadcastInDim(operandOp backends.Op, shape shapes.Shape, broadcastDims []int) (backends.Op, error) {
    // More flexible broadcast with dimension mapping
}
```

**Tasks:**
- [ ] Implement Broadcast using Tile/ExpandDims
- [ ] Implement BroadcastInDim
- [ ] Add tests

### 2.3 Iota (OpTypeIota)

Generate a tensor with values [0, 1, 2, ..., N-1] along an axis.

```go
// go-coreml/model/ops.go
func (b *Builder) Range(start, end, step *Value) *Value {
    // MIL operation "range_1d"
}

// gomlx/backends/coreml/ops.go
func (b *Builder) Iota(shape shapes.Shape, iotaDim int) (backends.Op, error) {
    // Use range_1d + broadcast to create iota tensor
}
```

**Tasks:**
- [ ] Implement Range in go-coreml
- [ ] Implement Iota wrapper
- [ ] Add tests

### 2.4 Additional Unary Operations

| OpType | MIL Op | Notes |
|--------|--------|-------|
| OpTypeRsqrt | `rsqrt` | 1/sqrt(x) |
| OpTypeExpm1 | N/A | exp(x) - 1, implement as Exp(x) - 1 |
| OpTypeLog1p | N/A | log(1 + x), implement as Log(Add(1, x)) |
| OpTypeIsFinite | `isfinite` | Check for finite values |
| OpTypeIsNaN | `isnan` | Check for NaN values |

**Tasks:**
- [ ] Add Rsqrt to go-coreml (MIL has `rsqrt`)
- [ ] Implement Rsqrt, Expm1, Log1p, IsFinite, IsNaN wrappers
- [ ] Add tests

### 2.5 Logical Operations

| OpType | MIL Op | Notes |
|--------|--------|-------|
| OpTypeLogicalAnd | `logical_and` | Boolean AND |
| OpTypeLogicalOr | `logical_or` | Boolean OR |
| OpTypeLogicalNot | `logical_not` | Boolean NOT |
| OpTypeLogicalXor | `logical_xor` | Boolean XOR |

**Tasks:**
- [ ] Add logical ops to go-coreml
- [ ] Add gomlx wrappers
- [ ] Add tests

### 2.6 Clamp (OpTypeClamp)

```go
func (b *Builder) Clamp(x, min, max backends.Op) (backends.Op, error) {
    // Implement as: Max(min, Min(max, x))
    // Or use MIL's clip operation
}
```

**Tasks:**
- [ ] Add Clip to go-coreml (MIL has `clip`)
- [ ] Implement Clamp wrapper
- [ ] Add tests

### 2.7 ConvertDType (OpTypeConvertDType)

```go
func (b *Builder) ConvertDType(x backends.Op, dtype dtypes.DType) (backends.Op, error) {
    // MIL operation "cast"
}
```

**Tasks:**
- [ ] Add Cast to go-coreml
- [ ] Implement ConvertDType wrapper
- [ ] Add tests for supported conversions

---

## Priority 3: Additional go-coreml Operations

### 3.1 L2 Normalization

```go
// go-coreml/model/ops.go
func (b *Builder) L2Norm(x *Value, axes []int64, epsilon float32) *Value {
    // MIL operation "l2_norm"
}
```

### 3.2 Linear (Fused MatMul + Bias)

```go
// go-coreml/model/ops.go
func (b *Builder) Linear(x, weight, bias *Value) *Value {
    // MIL operation "linear"
    // More efficient than separate matmul + add
}
```

### 3.3 Einsum

```go
// go-coreml/model/ops.go
func (b *Builder) Einsum(equation string, inputs []*Value) *Value {
    // MIL operation "einsum"
    // Powerful for attention and tensor contractions
}
```

**Tasks:**
- [ ] Implement L2Norm
- [ ] Implement Linear
- [ ] Research Einsum MIL support and implement
- [ ] Add tests

---

## Priority 4: ReduceWindow (Pooling via GoMLX)

GoMLX uses `ReduceWindow` for pooling operations rather than dedicated pool ops.

```go
func (b *Builder) ReduceWindow(
    operand backends.Op,
    reduceOpType ReduceOpType,  // Sum, Max, Min, Product
    windowDims []int,
    strides []int,
    paddingLow, paddingHigh []int,
) (backends.Op, error) {
    // Map to MaxPool/AvgPool for supported cases
    // Return error for unsupported configurations
}
```

**Tasks:**
- [ ] Implement ReduceWindow mapping to MaxPool/AvgPool
- [ ] Handle padding conversion
- [ ] Add tests

---

## Priority 5: Dynamic Operations

### 5.1 DynamicSlice

```go
func (b *Builder) DynamicSlice(
    operand backends.Op,
    startIndices []backends.Op,  // Runtime values
    sliceSizes []int,
) (backends.Op, error) {
    // CoreML's slice_by_index supports dynamic indices
}
```

### 5.2 DynamicUpdateSlice

```go
func (b *Builder) DynamicUpdateSlice(
    operand, update backends.Op,
    startIndices []backends.Op,
) (backends.Op, error) {
    // MIL operation "scatter" or "slice_update"
}
```

**Tasks:**
- [ ] Research MIL dynamic slice support
- [ ] Implement DynamicSlice
- [ ] Implement DynamicUpdateSlice
- [ ] Add tests

---

## Priority 6: Integration Testing

### 6.1 Simple CNN Model

Test end-to-end:
- Conv2D + BatchNorm + ReLU
- MaxPool
- Flatten + Dense

```go
func TestSimpleCNN(t *testing.T) {
    // Build model
    conv1 := Conv(input, weights1, ...)
    bn1 := BatchNorm(conv1, ...)
    relu1 := Max(bn1, Constant(0))
    pool1 := ReduceWindow(relu1, Max, [2,2], ...)
    // ... more layers
    // Verify output shape and reasonable values
}
```

### 6.2 Transformer Attention Block

Test:
- Q, K, V projections (DotGeneral)
- Attention scores (batched matmul)
- Softmax
- Weighted sum (batched matmul)
- Concatenation (multi-head)

```go
func TestAttentionBlock(t *testing.T) {
    // Multi-head self-attention
    // Q, K, V = Linear(input)
    // scores = Softmax(Q @ K^T / sqrt(d))
    // output = scores @ V
    // Verify shapes and numerics
}
```

### 6.3 Performance Benchmarks

Compare CoreML backend performance against:
- simplego backend (baseline)
- XLA CPU backend (if available)

**Metrics:**
- Compilation time
- Execution time
- Memory usage

**Tasks:**
- [ ] Implement CNN integration test
- [ ] Implement attention integration test
- [ ] Add performance benchmarks
- [ ] Document performance characteristics

---

## Implementation Order

### Sprint 1: Enhanced DotGeneral (1 week)
1. Add batch dimension support
2. Add arbitrary axis support via transpose
3. Comprehensive tests

### Sprint 2: Core Missing Wrappers (1 week)
1. Concatenate
2. Iota
3. Rsqrt, Expm1, Log1p
4. Clamp
5. ConvertDType

### Sprint 3: Logical and Broadcast Ops (0.5 weeks)
1. Logical ops (And, Or, Not, Xor)
2. Broadcast, BroadcastInDim
3. IsFinite, IsNaN

### Sprint 4: Additional MIL Ops (0.5 weeks)
1. L2Norm
2. Linear
3. Einsum (if MIL supports it)

### Sprint 5: ReduceWindow and Dynamic Ops (1 week)
1. ReduceWindow -> Pool mapping
2. DynamicSlice
3. DynamicUpdateSlice

### Sprint 6: Integration Testing (1 week)
1. CNN model test
2. Attention block test
3. Performance benchmarks
4. Documentation

---

## Success Criteria

- [ ] DotGeneral supports batch dimensions
- [ ] DotGeneral supports arbitrary contracting axes
- [ ] Concatenate working in gomlx backend
- [ ] All logical operations implemented
- [ ] Broadcast operations working
- [ ] ReduceWindow maps to pooling
- [ ] CNN model runs end-to-end
- [ ] Attention block runs end-to-end
- [ ] Performance benchmarks documented
- [ ] 90%+ of common ML operations covered

---

## Operations NOT Planned

These operations are low priority or not suitable for CoreML:

| Operation | Reason |
|-----------|--------|
| Bitwise ops | Rarely used in ML inference |
| Complex number ops | Limited CoreML support |
| Collective ops | CoreML is single-device |
| Sort | Limited use in inference |
| While loops | CoreML prefers unrolled graphs |
| Scatter ops | Complex, limited MIL support |
| FFT | Specialized, can add later if needed |

---

## Appendix: MIL Operations Reference

### Not Yet Implemented MIL Ops

| MIL Operation | Use Case |
|---------------|----------|
| `range_1d` | Iota implementation |
| `rsqrt` | Rsqrt |
| `clip` | Clamp |
| `cast` | ConvertDType |
| `logical_and/or/not/xor` | Logical ops |
| `isfinite`, `isnan` | Numeric checks |
| `l2_norm` | L2 normalization |
| `linear` | Fused dense layer |
| `einsum` | Einstein summation |
| `scatter` | Dynamic update |

### MIL Documentation

- https://apple.github.io/coremltools/docs-guides/source/ops-reference.html
- https://github.com/apple/coremltools/tree/main/coremltools/converters/mil/mil/ops
