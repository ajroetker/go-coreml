# Additional Tensor Operations for CoreML Backend - Phase 5

## Overview

This document details the implementation plan for the remaining operations needed to achieve comprehensive CoreML backend coverage. Phase 4 added 40+ operations. This phase focuses on:

1. Missing tensor manipulation ops (Concat, Tile, Pad, Reverse)
2. Convolution and Pooling operations
3. Normalization layers
4. Enhanced DotGeneral support
5. Broadcast operations

## Current State

**Implemented (Phases 3-4):**
- Unary: Abs, Neg, Exp, Log, Sqrt, Tanh, Logistic, Floor, Ceil, Round, Sign, Cos, Sin, Erf
- Binary: Add, Sub, Mul, Div, Pow, Max, Min
- Comparison: Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual
- Select: Where/Select
- Shape: Reshape, Transpose, Slice, Gather (partial), Squeeze*, ExpandDims*
- Reduction: ReduceSum, ReduceMax, ReduceMin, ReduceProd*, ArgMax*, ArgMin*
- Matrix: DotGeneral (simple matmul only)
- Activation: Gelu*, Silu*, LeakyRelu*, Elu*, Softplus*

*Only in go-coreml, not exposed in gomlx backend

**Target:** Full coverage for common ML model patterns (MLP, CNN, Transformer)

---

## Priority 1: Tensor Manipulation Ops

### 1.1 Concatenate

Join multiple tensors along an axis. Critical for residual connections and multi-head attention.

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Concatenate | `concat` | Takes tuple of values + axis |

**Challenge:** MIL's `concat` takes a tuple/list of values, not individual named arguments. This requires special handling in the serialization layer.

**Research Required:**
1. Investigate how MIL represents value tuples in the protobuf
2. Check if there's an alternative representation (e.g., variadic inputs)
3. Look at coremltools Python implementation for reference

**Implementation Plan:**

```go
// Option 1: Add tuple support to builder
// In model/builder.go - add new method for tuple arguments

type TupleArg struct {
    Values []*Value
}

func (b *Builder) addOpWithTuple(opType string, tupleArg []*Value, namedArgs map[string]*Value, name string, dtype DataType, shape []int64) *Value {
    // Create a special "tuple" argument binding
    // Serialize as a list of value references
}

// In model/ops.go
func (b *Builder) Concat(values []*Value, axis int64) *Value {
    if len(values) == 0 {
        panic("Concat requires at least one input")
    }
    if len(values) == 1 {
        return values[0] // No-op for single input
    }

    // Compute output shape
    outShape := make([]int64, len(values[0].shape))
    copy(outShape, values[0].shape)
    for i := 1; i < len(values); i++ {
        outShape[axis] += values[i].shape[axis]
    }

    axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})

    return b.addOpWithTuple("concat", values, map[string]*Value{
        "axis": axisVal,
    }, b.genName("concat"), values[0].dtype, outShape)
}
```

**Protobuf Investigation:**
```protobuf
// From MIL spec - need to verify exact format
message Operation {
    string type = 1;
    repeated Argument inputs = 2;
    repeated Output outputs = 3;
}

message Argument {
    oneof argument {
        ListValue list_value = 1;  // For tuple arguments
        NamedValue named_value = 2;
    }
}
```

**gomlx/backends/coreml/ops.go:**
```go
func (b *Builder) Concatenate(operands []backends.Op, axis int) (backends.Op, error) {
    opType := backends.OpTypeConcatenate

    // Validate and convert operands
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

    // Compute output shape
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
- [ ] Research MIL protobuf format for tuple/list arguments
- [ ] Add tuple argument support to go-coreml/model/builder.go
- [ ] Implement Concat in go-coreml/model/ops.go
- [ ] Add Concatenate to gomlx/backends/coreml/ops.go
- [ ] Add tests for 2, 3, and many tensor concatenation
- [ ] Test with different axes

---

### 1.2 Tile

Repeat a tensor along each axis.

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Tile | `tile` | Takes reps parameter |

**Implementation:**

```go
// go-coreml/model/ops.go
func (b *Builder) Tile(x *Value, reps []int64) *Value {
    // Compute output shape
    outShape := make([]int64, len(x.shape))
    for i := range outShape {
        outShape[i] = x.shape[i] * reps[i]
    }

    repsVal := b.Const(b.genName("reps"), Int32, []int64{int64(len(reps))}, toInt32Slice(reps))

    return b.addOp("tile", map[string]*Value{
        "x":    x,
        "reps": repsVal,
    }, b.genName("tile"), x.dtype, outShape)
}
```

**gomlx/backends/coreml/ops.go:**
```go
func (b *Builder) Tile(operandOp backends.Op, multiples []int) (backends.Op, error) {
    opType := backends.OpTypeTile
    inputs, err := b.checkOps(opType.String(), operandOp)
    if err != nil {
        return nil, err
    }
    operand := inputs[0]

    // Convert to int64
    multiplesInt64 := make([]int64, len(multiples))
    for i, m := range multiples {
        multiplesInt64[i] = int64(m)
    }

    outputShape, err := shapeinference.TileOp(operand.shape, multiples)
    if err != nil {
        return nil, err
    }

    resultValue := b.milBuilder.Tile(operand.milValue, multiplesInt64)
    node := b.newNode(opType, outputShape, resultValue, operand)

    return node, nil
}
```

**Tasks:**
- [ ] Implement Tile in go-coreml/model/ops.go
- [ ] Check if OpTypeTile exists in gomlx backends
- [ ] Add Tile to gomlx/backends/coreml/ops.go (if OpType exists)
- [ ] Add tests

---

### 1.3 Pad

Add padding to a tensor.

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Pad | `pad` | Supports constant, reflect, replicate modes |

**Implementation:**

```go
// go-coreml/model/ops.go

// PadMode represents padding mode
type PadMode int

const (
    PadConstant PadMode = iota
    PadReflect
    PadReplicate
)

func (b *Builder) Pad(x *Value, padBefore, padAfter []int64, mode PadMode, constantValue float32) *Value {
    // Compute output shape
    outShape := make([]int64, len(x.shape))
    for i := range outShape {
        outShape[i] = x.shape[i] + padBefore[i] + padAfter[i]
    }

    // MIL pad format: [before_0, after_0, before_1, after_1, ...]
    padSpec := make([]int32, len(padBefore)*2)
    for i := range padBefore {
        padSpec[i*2] = int32(padBefore[i])
        padSpec[i*2+1] = int32(padAfter[i])
    }

    padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)

    inputs := map[string]*Value{
        "x":   x,
        "pad": padVal,
    }

    // Add mode-specific parameters
    switch mode {
    case PadConstant:
        modeVal := b.Const(b.genName("mode"), String, []int64{}, "constant")
        constVal := b.Const(b.genName("constant_val"), x.dtype, []int64{}, []float32{constantValue})
        inputs["mode"] = modeVal
        inputs["constant_val"] = constVal
    case PadReflect:
        modeVal := b.Const(b.genName("mode"), String, []int64{}, "reflect")
        inputs["mode"] = modeVal
    case PadReplicate:
        modeVal := b.Const(b.genName("mode"), String, []int64{}, "replicate")
        inputs["mode"] = modeVal
    }

    return b.addOp("pad", inputs, b.genName("pad"), x.dtype, outShape)
}
```

**gomlx/backends/coreml/ops.go:**
```go
func (b *Builder) Pad(operandOp backends.Op, low, high, interior []int) (backends.Op, error) {
    opType := backends.OpTypePad
    inputs, err := b.checkOps(opType.String(), operandOp)
    if err != nil {
        return nil, err
    }
    operand := inputs[0]

    // GoMLX Pad also has interior padding (between elements)
    // CoreML doesn't support interior padding directly
    for _, i := range interior {
        if i != 0 {
            return nil, errors.Errorf("Pad: CoreML backend does not support interior padding")
        }
    }

    // Convert to int64
    lowInt64 := make([]int64, len(low))
    highInt64 := make([]int64, len(high))
    for i := range low {
        lowInt64[i] = int64(low[i])
        highInt64[i] = int64(high[i])
    }

    outputShape, err := shapeinference.PadOp(operand.shape, low, high, interior)
    if err != nil {
        return nil, err
    }

    resultValue := b.milBuilder.Pad(operand.milValue, lowInt64, highInt64, model.PadConstant, 0.0)
    node := b.newNode(opType, outputShape, resultValue, operand)

    return node, nil
}
```

**Tasks:**
- [ ] Implement Pad in go-coreml/model/ops.go with all modes
- [ ] Check if OpTypePad exists in gomlx backends
- [ ] Add Pad to gomlx/backends/coreml/ops.go (handle interior padding limitation)
- [ ] Add tests for each padding mode

---

### 1.4 Reverse

Reverse tensor along specified axes.

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Reverse | `reverse` | Reverses along specified axes |

**Implementation:**

```go
// go-coreml/model/ops.go
func (b *Builder) Reverse(x *Value, axes []int64) *Value {
    axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))

    return b.addOp("reverse", map[string]*Value{
        "x":    x,
        "axes": axesVal,
    }, b.genName("reverse"), x.dtype, x.shape) // Shape unchanged
}
```

**gomlx/backends/coreml/ops.go:**
```go
func (b *Builder) Reverse(operandOp backends.Op, axes []int) (backends.Op, error) {
    opType := backends.OpTypeReverse
    inputs, err := b.checkOps(opType.String(), operandOp)
    if err != nil {
        return nil, err
    }
    operand := inputs[0]

    axesInt64 := make([]int64, len(axes))
    for i, a := range axes {
        axesInt64[i] = int64(a)
    }

    // Output shape is same as input
    resultValue := b.milBuilder.Reverse(operand.milValue, axesInt64)
    node := b.newNode(opType, operand.shape, resultValue, operand)

    return node, nil
}
```

**Tasks:**
- [ ] Implement Reverse in go-coreml/model/ops.go
- [ ] Check if OpTypeReverse exists in gomlx backends
- [ ] Add Reverse to gomlx/backends/coreml/ops.go
- [ ] Add tests

---

## Priority 2: Convolution Operations

Critical for CNN models.

### 2.1 Conv2D

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| ConvGeneral | `conv` | General N-D convolution |

**MIL Conv Parameters:**
- `x`: Input tensor [N, C_in, H, W] (NCHW format)
- `weight`: Filter tensor [C_out, C_in/groups, kH, kW]
- `strides`: [stride_h, stride_w]
- `pad_type`: "same", "valid", or "custom"
- `pad`: Custom padding [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
- `dilations`: [dilation_h, dilation_w]
- `groups`: Number of groups for grouped convolution

**Implementation:**

```go
// go-coreml/model/ops.go

// ConvPadType represents convolution padding type
type ConvPadType int

const (
    ConvPadValid ConvPadType = iota
    ConvPadSame
    ConvPadCustom
)

func (b *Builder) Conv(
    x, weight *Value,
    strides, dilations []int64,
    padType ConvPadType,
    padBefore, padAfter []int64, // Only used if padType == ConvPadCustom
    groups int64,
) *Value {
    // Validate dimensions
    // x: [N, C_in, H, W] for 2D conv
    // weight: [C_out, C_in/groups, kH, kW]

    xShape := x.shape
    wShape := weight.shape

    N := xShape[0]
    C_out := wShape[0]

    // Compute output spatial dimensions
    // H_out = (H_in + pad_h_before + pad_h_after - dilation_h * (kH - 1) - 1) / stride_h + 1

    var H_out, W_out int64
    switch padType {
    case ConvPadSame:
        H_out = (xShape[2] + strides[0] - 1) / strides[0]
        W_out = (xShape[3] + strides[1] - 1) / strides[1]
    case ConvPadValid:
        kH := wShape[2]
        kW := wShape[3]
        H_out = (xShape[2] - dilations[0]*(kH-1) - 1) / strides[0] + 1
        W_out = (xShape[3] - dilations[1]*(kW-1) - 1) / strides[1] + 1
    case ConvPadCustom:
        kH := wShape[2]
        kW := wShape[3]
        H_out = (xShape[2] + padBefore[0] + padAfter[0] - dilations[0]*(kH-1) - 1) / strides[0] + 1
        W_out = (xShape[3] + padBefore[1] + padAfter[1] - dilations[1]*(kW-1) - 1) / strides[1] + 1
    }

    outShape := []int64{N, C_out, H_out, W_out}

    // Build arguments
    stridesVal := b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))
    dilationsVal := b.Const(b.genName("dilations"), Int32, []int64{int64(len(dilations))}, toInt32Slice(dilations))
    groupsVal := b.Const(b.genName("groups"), Int32, []int64{}, []int32{int32(groups)})

    inputs := map[string]*Value{
        "x":         x,
        "weight":    weight,
        "strides":   stridesVal,
        "dilations": dilationsVal,
        "groups":    groupsVal,
    }

    switch padType {
    case ConvPadSame:
        padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "same")
        inputs["pad_type"] = padTypeVal
    case ConvPadValid:
        padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "valid")
        inputs["pad_type"] = padTypeVal
    case ConvPadCustom:
        padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "custom")
        // Combine before/after: [h_before, h_after, w_before, w_after]
        padSpec := make([]int32, len(padBefore)*2)
        for i := range padBefore {
            padSpec[i*2] = int32(padBefore[i])
            padSpec[i*2+1] = int32(padAfter[i])
        }
        padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
        inputs["pad_type"] = padTypeVal
        inputs["pad"] = padVal
    }

    return b.addOp("conv", inputs, b.genName("conv"), x.dtype, outShape)
}

// ConvWithBias adds bias after convolution
func (b *Builder) ConvWithBias(
    x, weight, bias *Value,
    strides, dilations []int64,
    padType ConvPadType,
    padBefore, padAfter []int64,
    groups int64,
) *Value {
    conv := b.Conv(x, weight, strides, dilations, padType, padBefore, padAfter, groups)
    // Reshape bias for broadcasting: [C_out] -> [1, C_out, 1, 1]
    biasReshaped := b.Reshape(bias, []int64{1, bias.shape[0], 1, 1})
    return b.Add(conv, biasReshaped)
}
```

**gomlx/backends/coreml/ops.go:**

GoMLX uses `ConvGeneralDilated` with dimension numbers and feature group count.

```go
func (b *Builder) ConvGeneralDilated(
    operandOp, kernelOp backends.Op,
    strides, dilations, paddingLow, paddingHigh []int,
    inputBatchDim, inputFeatureDim int,
    inputSpatialDims []int,
    kernelInputFeatureDim, kernelOutputFeatureDim int,
    kernelSpatialDims []int,
    outputBatchDim, outputFeatureDim int,
    outputSpatialDims []int,
    featureGroupCount, batchGroupCount int,
) (backends.Op, error) {
    opType := backends.OpTypeConvGeneralDilated

    inputs, err := b.checkOps(opType.String(), operandOp, kernelOp)
    if err != nil {
        return nil, err
    }
    operand, kernel := inputs[0], inputs[1]

    // Validate dimension numbers
    // CoreML expects NCHW format for both input and output
    // Need to transpose if format differs

    if inputBatchDim != 0 || inputFeatureDim != 1 {
        return nil, errors.Errorf("ConvGeneralDilated: CoreML requires NCHW format (batch=0, feature=1)")
    }

    // Convert parameters
    stridesInt64 := toInt64Slice(strides)
    dilationsInt64 := toInt64Slice(dilations)

    // Determine padding type
    var padType model.ConvPadType
    var padBefore, padAfter []int64

    allZero := true
    for i := range paddingLow {
        if paddingLow[i] != 0 || paddingHigh[i] != 0 {
            allZero = false
            break
        }
    }

    if allZero {
        padType = model.ConvPadValid
    } else {
        padType = model.ConvPadCustom
        padBefore = toInt64Slice(paddingLow)
        padAfter = toInt64Slice(paddingHigh)
    }

    // Compute output shape using shapeinference
    outputShape, err := shapeinference.ConvGeneralDilatedOp(...)
    if err != nil {
        return nil, err
    }

    resultValue := b.milBuilder.Conv(
        operand.milValue, kernel.milValue,
        stridesInt64, dilationsInt64,
        padType, padBefore, padAfter,
        int64(featureGroupCount),
    )

    node := b.newNode(opType, outputShape, resultValue, operand, kernel)
    return node, nil
}
```

**Tasks:**
- [ ] Research MIL conv parameter format in detail
- [ ] Implement Conv in go-coreml/model/ops.go
- [ ] Add ConvTranspose for deconvolution
- [ ] Add ConvGeneralDilated to gomlx/backends/coreml/ops.go
- [ ] Handle NHWC to NCHW format conversion if needed
- [ ] Add tests for various conv configurations

---

### 2.2 Conv Transpose (Deconvolution)

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| ConvTranspose | `conv_transpose` | Transposed convolution |

**Implementation:**

```go
// go-coreml/model/ops.go
func (b *Builder) ConvTranspose(
    x, weight *Value,
    strides, dilations []int64,
    padType ConvPadType,
    padBefore, padAfter []int64,
    outputPadding []int64, // Additional padding for output shape
    groups int64,
) *Value {
    // Similar to Conv but output shape calculation is different
    // H_out = (H_in - 1) * stride - 2*pad + dilation*(kH - 1) + output_padding + 1

    // ... implementation
}
```

**Tasks:**
- [ ] Research MIL conv_transpose parameters
- [ ] Implement ConvTranspose in go-coreml
- [ ] Add to gomlx backend if ConvTranspose op exists

---

## Priority 3: Pooling Operations

### 3.1 MaxPool and AvgPool

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| MaxPool | `max_pool` | Max pooling |
| AvgPool | `avg_pool` | Average pooling |

**Implementation:**

```go
// go-coreml/model/ops.go
func (b *Builder) MaxPool(
    x *Value,
    kernelSize, strides []int64,
    padType ConvPadType,
    padBefore, padAfter []int64,
) *Value {
    // Compute output shape similar to conv
    // ...

    kernelVal := b.Const(b.genName("kernel_sizes"), Int32, []int64{int64(len(kernelSize))}, toInt32Slice(kernelSize))
    stridesVal := b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))

    inputs := map[string]*Value{
        "x":            x,
        "kernel_sizes": kernelVal,
        "strides":      stridesVal,
    }

    // Add padding parameters...

    return b.addOp("max_pool", inputs, b.genName("max_pool"), x.dtype, outShape)
}

func (b *Builder) AvgPool(
    x *Value,
    kernelSize, strides []int64,
    padType ConvPadType,
    padBefore, padAfter []int64,
    excludePaddingFromAverage bool,
) *Value {
    // Similar to MaxPool
    // ...

    excludeVal := b.Const(b.genName("exclude_padding"), Bool, []int64{}, []bool{excludePaddingFromAverage})
    inputs["exclude_padding_from_average"] = excludeVal

    return b.addOp("avg_pool", inputs, b.genName("avg_pool"), x.dtype, outShape)
}
```

**Tasks:**
- [ ] Implement MaxPool in go-coreml
- [ ] Implement AvgPool in go-coreml
- [ ] Add to gomlx backend (check for OpType)
- [ ] Add tests

### 3.2 Global Pooling

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| GlobalAvgPool | `reduce_mean` | Use reduce_mean on spatial dims |
| GlobalMaxPool | `reduce_max` | Use reduce_max on spatial dims |

Global pooling can be implemented using existing reduce operations:

```go
// Helper functions in ops.go
func (b *Builder) GlobalAvgPool2D(x *Value) *Value {
    // Reduce over H, W dimensions (axes 2, 3 for NCHW)
    return b.ReduceMean(x, []int64{2, 3}, true) // keepDims=true
}

func (b *Builder) GlobalMaxPool2D(x *Value) *Value {
    return b.ReduceMax(x, []int64{2, 3}, true)
}
```

**Tasks:**
- [ ] Add GlobalAvgPool2D, GlobalMaxPool2D convenience functions
- [ ] Add tests

---

## Priority 4: Normalization Layers

### 4.1 Batch Normalization

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| BatchNorm | `batch_norm` | Batch normalization |

**Implementation:**

```go
// go-coreml/model/ops.go
func (b *Builder) BatchNorm(
    x, mean, variance, gamma, beta *Value,
    epsilon float32,
) *Value {
    epsVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

    return b.addOp("batch_norm", map[string]*Value{
        "x":        x,
        "mean":     mean,
        "variance": variance,
        "gamma":    gamma,
        "beta":     beta,
        "epsilon":  epsVal,
    }, b.genName("batch_norm"), x.dtype, x.shape)
}
```

**Tasks:**
- [ ] Implement BatchNorm in go-coreml
- [ ] Check gomlx BatchNorm interface
- [ ] Add tests

### 4.2 Layer Normalization

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| LayerNorm | `layer_norm` | Layer normalization |

**Implementation:**

```go
// go-coreml/model/ops.go
func (b *Builder) LayerNorm(
    x, gamma, beta *Value,
    axes []int64,
    epsilon float32,
) *Value {
    axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
    epsVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

    return b.addOp("layer_norm", map[string]*Value{
        "x":       x,
        "gamma":   gamma,
        "beta":    beta,
        "axes":    axesVal,
        "epsilon": epsVal,
    }, b.genName("layer_norm"), x.dtype, x.shape)
}
```

**Tasks:**
- [ ] Implement LayerNorm in go-coreml
- [ ] Check gomlx LayerNorm interface
- [ ] Add tests

### 4.3 Instance Normalization

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| InstanceNorm | `instance_norm` | Instance normalization |

**Implementation:**

```go
// go-coreml/model/ops.go
func (b *Builder) InstanceNorm(
    x, gamma, beta *Value,
    epsilon float32,
) *Value {
    epsVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

    return b.addOp("instance_norm", map[string]*Value{
        "x":       x,
        "gamma":   gamma,
        "beta":    beta,
        "epsilon": epsVal,
    }, b.genName("instance_norm"), x.dtype, x.shape)
}
```

**Tasks:**
- [ ] Implement InstanceNorm in go-coreml
- [ ] Add tests

---

## Priority 5: Enhanced DotGeneral

The current DotGeneral only supports simple matrix multiplication. We need to support:
- Batch dimensions
- Arbitrary contracting axes
- Transposed inputs

**Implementation Strategy:**

Use Transpose and Reshape to normalize inputs to standard matmul format.

```go
// gomlx/backends/coreml/ops.go

func (b *Builder) DotGeneral(
    lhsOp backends.Op,
    lhsContractingAxes, lhsBatchAxes []int,
    rhsOp backends.Op,
    rhsContractingAxes, rhsBatchAxes []int,
) (backends.Op, error) {
    // ... existing validation ...

    // For complex cases, normalize to standard batched matmul format:
    // lhs: [B1, B2, ..., M, K]
    // rhs: [B1, B2, ..., K, N]
    // out: [B1, B2, ..., M, N]

    // Step 1: Transpose to move batch dims first, then M/K
    lhsTransposed := transposeToBatchedMatmul(lhs, lhsBatchAxes, lhsContractingAxes)
    rhsTransposed := transposeToBatchedMatmul(rhs, rhsBatchAxes, rhsContractingAxes)

    // Step 2: Reshape to merge batch dimensions if needed
    // ...

    // Step 3: Call batched matmul
    // ...

    // Step 4: Reshape/transpose output to expected shape
    // ...
}
```

**Tasks:**
- [ ] Implement helper functions for normalizing DotGeneral inputs
- [ ] Support batch dimensions
- [ ] Support arbitrary contracting axes via transpose
- [ ] Add comprehensive tests for various DotGeneral configurations

---

## Priority 6: Broadcast Operations

### 6.1 BroadcastTo

Explicitly broadcast a tensor to a larger shape.

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| BroadcastTo | Could use `tile` or native broadcast | Explicit broadcasting |

**Implementation:**

CoreML handles broadcasting implicitly in most operations. For explicit broadcast, we can use Tile or a combination of ExpandDims and Tile.

```go
// go-coreml/model/ops.go
func (b *Builder) BroadcastTo(x *Value, targetShape []int64) *Value {
    // Option 1: Use tile to broadcast
    // Calculate reps needed for each dimension

    xRank := len(x.shape)
    targetRank := len(targetShape)

    // First expand dims if needed
    if xRank < targetRank {
        newAxes := make([]int64, targetRank-xRank)
        for i := range newAxes {
            newAxes[i] = int64(i)
        }
        x = b.ExpandDims(x, newAxes)
    }

    // Then tile to match target shape
    reps := make([]int64, len(targetShape))
    for i := range reps {
        if x.shape[i] == 1 && targetShape[i] > 1 {
            reps[i] = targetShape[i]
        } else {
            reps[i] = 1
        }
    }

    return b.Tile(x, reps)
}
```

**Tasks:**
- [ ] Implement BroadcastTo in go-coreml
- [ ] Check if gomlx has explicit broadcast op
- [ ] Add tests

---

## Implementation Order

### Sprint 1: Tensor Manipulation (1 week)
1. Research MIL tuple format for Concat
2. Implement Tile, Pad, Reverse in go-coreml
3. Add gomlx wrappers for ops with defined OpTypes
4. Tests for each operation

### Sprint 2: Convolution (1 week)
1. Research MIL conv parameters thoroughly
2. Implement Conv2D with all padding modes
3. Implement ConvTranspose
4. Add gomlx ConvGeneralDilated wrapper
5. Tests

### Sprint 3: Pooling (0.5 weeks)
1. Implement MaxPool, AvgPool
2. Add global pooling helpers
3. Add gomlx wrappers
4. Tests

### Sprint 4: Normalization (0.5 weeks)
1. Implement BatchNorm, LayerNorm, InstanceNorm
2. Add gomlx wrappers
3. Tests

### Sprint 5: Enhanced DotGeneral (1 week)
1. Implement transpose/reshape normalization
2. Support batch dimensions
3. Comprehensive tests

### Sprint 6: Concat & Cleanup (0.5 weeks)
1. Implement Concat with tuple support
2. Code cleanup and documentation
3. Integration tests

---

## Testing Strategy

1. **Unit tests per operation**: Test basic functionality with known inputs/outputs
2. **Shape inference tests**: Verify output shapes are computed correctly
3. **Edge cases**: Empty tensors, scalars, single-element tensors
4. **Numerical accuracy**: Compare against simplego or XLA backend
5. **Integration tests**: Build and run simple models:
   - MLP with Tile for batch broadcast
   - Simple CNN (Conv + Pool + Norm)
   - Attention mechanism (DotGeneral + Concatenate)

---

## Success Criteria

- [x] All tensor manipulation ops implemented (Concat, Tile, Pad, Reverse)
- [x] Convolution operations working for 2D case
- [x] Pooling operations working
- [x] Normalization layers working
- [ ] DotGeneral supports batch dimensions (future work)
- [x] All tests passing
- [ ] Can run simple CNN model end-to-end (integration test pending)
- [ ] Can run transformer attention block (integration test pending)

---

## Phase 5 Implementation Notes (January 2026)

### Completed Operations

**go-coreml/model/ops.go:**
1. **Tile** - Repeat tensor along each axis
2. **Pad** - Add padding with constant/reflect/replicate modes
3. **Reverse** - Reverse along specified axes
4. **Concat** - Concatenate multiple tensors (required adding list argument support to builder)
5. **Conv** - 2D convolution with all padding modes
6. **ConvTranspose** - Transposed/deconvolution
7. **ConvWithBias** - Convenience helper
8. **MaxPool** - Max pooling with all padding modes
9. **AvgPool** - Average pooling with exclude_padding option
10. **GlobalAvgPool2D** - Global average pooling (via ReduceMean)
11. **GlobalMaxPool2D** - Global max pooling (via ReduceMax)
12. **BatchNorm** - Batch normalization
13. **LayerNorm** - Layer normalization
14. **InstanceNorm** - Instance normalization

**gomlx/backends/coreml/ops.go:**
1. **Pad** (OpTypePad) - Constant padding only, no interior padding
2. **Reverse** (OpTypeReverse) - Full support
3. **ConvGeneral** (OpTypeConvGeneral) - NCHW layout, standard convolutions
4. **BatchNormForInference** (OpTypeBatchNormForInference) - Feature axis=1

### Key Technical Decisions

1. **Concat Tuple Support**: MIL's concat requires tuple/list arguments. Added `addOpWithListArg()` to builder.go to handle this pattern.

2. **Padding Type Constants**: Shared `ConvPadType` enum (Valid/Same/Custom) between convolution and pooling operations.

3. **CoreML Limitations**:
   - Pad: No interior padding support (use XLA for this)
   - ConvGeneral: NCHW layout required, no batch group count
   - BatchNorm: Feature axis must be 1

### Test Coverage

- go-coreml: All model tests pass (35+ tests)
- gomlx/backends/coreml: All tests pass (23 tests including new Pad, Reverse, ConvGeneral, BatchNorm tests)

### Remaining Work

- Enhanced DotGeneral with batch dimensions (Priority 5)
- Integration tests for full CNN and transformer models
- BroadcastTo operation (can use Tile + ExpandDims as workaround)
- L2 normalization
- Linear (fused matmul + bias)
- Einsum

---

## Appendix: MIL Operation Reference

Useful MIL documentation:
- https://apple.github.io/coremltools/docs-guides/source/ops-reference.html
- https://github.com/apple/coremltools/tree/main/coremltools/converters/mil/mil/ops

Key MIL ops implemented in this phase:
- `concat` - Concatenate tensors ✓
- `tile` - Tile/repeat tensor ✓
- `pad` - Pad tensor ✓
- `reverse` - Reverse along axes ✓
- `conv` - Convolution ✓
- `conv_transpose` - Transposed convolution ✓
- `max_pool` - Max pooling ✓
- `avg_pool` - Average pooling ✓
- `batch_norm` - Batch normalization ✓
- `layer_norm` - Layer normalization ✓
- `instance_norm` - Instance normalization ✓

Remaining MIL ops:
- `l2_norm` - L2 normalization
- `linear` - Linear/dense layer (fused matmul + bias)
- `einsum` - Einstein summation
