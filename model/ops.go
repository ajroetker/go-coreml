package model

// This file contains MIL operation builders.
// MIL operations are documented at:
// https://apple.github.io/coremltools/docs-guides/source/ops-reference.html

// ConvPadType represents convolution/pooling padding type.
type ConvPadType int

const (
	// ConvPadValid means no padding (only valid positions).
	ConvPadValid ConvPadType = iota
	// ConvPadSame means output size equals input size (with stride=1).
	ConvPadSame
	// ConvPadCustom means custom padding specified by padBefore and padAfter.
	ConvPadCustom
)

// Add performs element-wise addition: z = x + y.
func (b *Builder) Add(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("add", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("add"), x.dtype, outShape)
}

// Sub performs element-wise subtraction: z = x - y.
func (b *Builder) Sub(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("sub", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("sub"), x.dtype, outShape)
}

// Mul performs element-wise multiplication: z = x * y.
func (b *Builder) Mul(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("mul", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("mul"), x.dtype, outShape)
}

// Div performs element-wise division: z = x / y.
func (b *Builder) Div(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("real_div", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("div"), x.dtype, outShape)
}

// MatMul performs matrix multiplication: z = x @ y.
// x: [..., M, K], y: [..., K, N] -> z: [..., M, N]
func (b *Builder) MatMul(x, y *Value) *Value {
	return b.MatMulTranspose(x, y, false, false)
}

// MatMulTranspose performs matrix multiplication with optional transposes.
// x: [..., M, K], y: [..., K, N] -> z: [..., M, N]
// If transposeX is true, x is transposed before multiplication.
// If transposeY is true, y is transposed before multiplication.
func (b *Builder) MatMulTranspose(x, y *Value, transposeX, transposeY bool) *Value {
	// Compute output shape for matmul
	xShape := x.shape
	yShape := y.shape

	// Adjust shapes based on transposes
	xM := xShape[len(xShape)-2]
	xK := xShape[len(xShape)-1]
	yK := yShape[len(yShape)-2]
	yN := yShape[len(yShape)-1]

	if transposeX {
		xM, xK = xK, xM
	}
	if transposeY {
		yK, yN = yN, yK
	}
	_ = xK // K dimension should match
	_ = yK

	outShape := make([]int64, len(xShape))
	copy(outShape, xShape[:len(xShape)-2])
	outShape[len(outShape)-2] = xM
	outShape[len(outShape)-1] = yN

	transposeXVal := b.Const(b.genName("transpose_x"), Bool, []int64{}, []bool{transposeX})
	transposeYVal := b.Const(b.genName("transpose_y"), Bool, []int64{}, []bool{transposeY})

	return b.addOp("matmul", map[string]*Value{
		"x":           x,
		"y":           y,
		"transpose_x": transposeXVal,
		"transpose_y": transposeYVal,
	}, b.genName("matmul"), x.dtype, outShape)
}

// Relu applies rectified linear unit: z = max(x, 0).
func (b *Builder) Relu(x *Value) *Value {
	return b.addOp("relu", map[string]*Value{
		"x": x,
	}, b.genName("relu"), x.dtype, x.shape)
}

// Sigmoid applies sigmoid activation: z = 1 / (1 + exp(-x)).
func (b *Builder) Sigmoid(x *Value) *Value {
	return b.addOp("sigmoid", map[string]*Value{
		"x": x,
	}, b.genName("sigmoid"), x.dtype, x.shape)
}

// Tanh applies hyperbolic tangent: z = tanh(x).
func (b *Builder) Tanh(x *Value) *Value {
	return b.addOp("tanh", map[string]*Value{
		"x": x,
	}, b.genName("tanh"), x.dtype, x.shape)
}

// Softmax applies softmax along the specified axis.
func (b *Builder) Softmax(x *Value, axis int) *Value {
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	return b.addOp("softmax", map[string]*Value{
		"x":    x,
		"axis": axisVal,
	}, b.genName("softmax"), x.dtype, x.shape)
}

// Exp computes element-wise exponential: z = exp(x).
func (b *Builder) Exp(x *Value) *Value {
	return b.addOp("exp", map[string]*Value{
		"x": x,
	}, b.genName("exp"), x.dtype, x.shape)
}

// Log computes element-wise natural logarithm: z = log(x).
func (b *Builder) Log(x *Value) *Value {
	return b.addOp("log", map[string]*Value{
		"x": x,
	}, b.genName("log"), x.dtype, x.shape)
}

// Sqrt computes element-wise square root: z = sqrt(x).
func (b *Builder) Sqrt(x *Value) *Value {
	return b.addOp("sqrt", map[string]*Value{
		"x": x,
	}, b.genName("sqrt"), x.dtype, x.shape)
}

// Neg computes element-wise negation: z = -x.
func (b *Builder) Neg(x *Value) *Value {
	return b.addOp("neg", map[string]*Value{
		"x": x,
	}, b.genName("neg"), x.dtype, x.shape)
}

// Abs computes element-wise absolute value: z = |x|.
func (b *Builder) Abs(x *Value) *Value {
	return b.addOp("abs", map[string]*Value{
		"x": x,
	}, b.genName("abs"), x.dtype, x.shape)
}

// Pow performs element-wise power: z = x^y.
func (b *Builder) Pow(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("pow", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("pow"), x.dtype, outShape)
}

// Maximum computes element-wise maximum: z = max(x, y).
func (b *Builder) Maximum(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("maximum", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("maximum"), x.dtype, outShape)
}

// Minimum computes element-wise minimum: z = min(x, y).
func (b *Builder) Minimum(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("minimum", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("minimum"), x.dtype, outShape)
}

// Floor computes element-wise floor: z = floor(x).
func (b *Builder) Floor(x *Value) *Value {
	return b.addOp("floor", map[string]*Value{
		"x": x,
	}, b.genName("floor"), x.dtype, x.shape)
}

// Ceil computes element-wise ceiling: z = ceil(x).
func (b *Builder) Ceil(x *Value) *Value {
	return b.addOp("ceil", map[string]*Value{
		"x": x,
	}, b.genName("ceil"), x.dtype, x.shape)
}

// Round computes element-wise rounding: z = round(x).
func (b *Builder) Round(x *Value) *Value {
	return b.addOp("round", map[string]*Value{
		"x": x,
	}, b.genName("round"), x.dtype, x.shape)
}

// Sign computes element-wise sign: z = sign(x).
// Returns -1 for negative values, 0 for zero, and 1 for positive values.
func (b *Builder) Sign(x *Value) *Value {
	return b.addOp("sign", map[string]*Value{
		"x": x,
	}, b.genName("sign"), x.dtype, x.shape)
}

// Cos computes element-wise cosine: z = cos(x).
func (b *Builder) Cos(x *Value) *Value {
	return b.addOp("cos", map[string]*Value{
		"x": x,
	}, b.genName("cos"), x.dtype, x.shape)
}

// Sin computes element-wise sine: z = sin(x).
func (b *Builder) Sin(x *Value) *Value {
	return b.addOp("sin", map[string]*Value{
		"x": x,
	}, b.genName("sin"), x.dtype, x.shape)
}

// Acos computes element-wise arc cosine: z = acos(x).
func (b *Builder) Acos(x *Value) *Value {
	return b.addOp("acos", map[string]*Value{
		"x": x,
	}, b.genName("acos"), x.dtype, x.shape)
}

// Asin computes element-wise arc sine: z = asin(x).
func (b *Builder) Asin(x *Value) *Value {
	return b.addOp("asin", map[string]*Value{
		"x": x,
	}, b.genName("asin"), x.dtype, x.shape)
}

// Atan computes element-wise arc tangent: z = atan(x).
func (b *Builder) Atan(x *Value) *Value {
	return b.addOp("atan", map[string]*Value{
		"x": x,
	}, b.genName("atan"), x.dtype, x.shape)
}

// Cosh computes element-wise hyperbolic cosine: z = cosh(x).
func (b *Builder) Cosh(x *Value) *Value {
	return b.addOp("cosh", map[string]*Value{
		"x": x,
	}, b.genName("cosh"), x.dtype, x.shape)
}

// Sinh computes element-wise hyperbolic sine: z = sinh(x).
func (b *Builder) Sinh(x *Value) *Value {
	return b.addOp("sinh", map[string]*Value{
		"x": x,
	}, b.genName("sinh"), x.dtype, x.shape)
}

// Erf computes element-wise error function: z = erf(x).
func (b *Builder) Erf(x *Value) *Value {
	return b.addOp("erf", map[string]*Value{
		"x": x,
	}, b.genName("erf"), x.dtype, x.shape)
}

// Gelu computes Gaussian Error Linear Unit: x * Φ(x) where Φ is the cumulative distribution
// function of the standard normal distribution.
// mode should be "EXACT" or "TANH_APPROXIMATION".
func (b *Builder) Gelu(x *Value, mode string) *Value {
	modeVal := b.Const(b.genName("mode"), String, []int64{}, mode)
	return b.addOp("gelu", map[string]*Value{
		"x":    x,
		"mode": modeVal,
	}, b.genName("gelu"), x.dtype, x.shape)
}

// Silu computes Sigmoid Linear Unit (Swish): x * sigmoid(x).
func (b *Builder) Silu(x *Value) *Value {
	return b.addOp("silu", map[string]*Value{
		"x": x,
	}, b.genName("silu"), x.dtype, x.shape)
}

// LeakyRelu computes Leaky ReLU: max(x, alpha*x) where alpha is typically small (e.g., 0.01).
func (b *Builder) LeakyRelu(x *Value, alpha float32) *Value {
	alphaVal := b.Const(b.genName("alpha"), Float32, []int64{}, []float32{alpha})
	return b.addOp("leaky_relu", map[string]*Value{
		"x":     x,
		"alpha": alphaVal,
	}, b.genName("leaky_relu"), x.dtype, x.shape)
}

// Elu computes Exponential Linear Unit: x if x > 0, else alpha * (exp(x) - 1).
func (b *Builder) Elu(x *Value, alpha float32) *Value {
	alphaVal := b.Const(b.genName("alpha"), Float32, []int64{}, []float32{alpha})
	return b.addOp("elu", map[string]*Value{
		"x":     x,
		"alpha": alphaVal,
	}, b.genName("elu"), x.dtype, x.shape)
}

// Softplus computes smooth approximation of ReLU: log(1 + exp(x)).
func (b *Builder) Softplus(x *Value) *Value {
	return b.addOp("softplus", map[string]*Value{
		"x": x,
	}, b.genName("softplus"), x.dtype, x.shape)
}

// Reshape changes the shape of a tensor.
func (b *Builder) Reshape(x *Value, shape []int64) *Value {
	shapeVal := b.Const(b.genName("shape"), Int32, []int64{int64(len(shape))}, toInt32Slice(shape))
	return b.addOp("reshape", map[string]*Value{
		"x":     x,
		"shape": shapeVal,
	}, b.genName("reshape"), x.dtype, shape)
}

// Transpose permutes the dimensions of a tensor.
func (b *Builder) Transpose(x *Value, perm []int64) *Value {
	permVal := b.Const(b.genName("perm"), Int32, []int64{int64(len(perm))}, toInt32Slice(perm))

	// Compute output shape
	outShape := make([]int64, len(perm))
	for i, p := range perm {
		outShape[i] = x.shape[p]
	}

	return b.addOp("transpose", map[string]*Value{
		"x":    x,
		"perm": permVal,
	}, b.genName("transpose"), x.dtype, outShape)
}

// ReduceSum computes sum along specified axes.
func (b *Builder) ReduceSum(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_sum", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_sum"), x.dtype, outShape)
}

// ReduceMean computes mean along specified axes.
func (b *Builder) ReduceMean(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_mean", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_mean"), x.dtype, outShape)
}

// ReduceMax computes max along specified axes.
func (b *Builder) ReduceMax(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_max", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_max"), x.dtype, outShape)
}

// ReduceMin computes min along specified axes.
func (b *Builder) ReduceMin(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_min", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_min"), x.dtype, outShape)
}

// ReduceProd computes product along specified axes.
func (b *Builder) ReduceProd(x *Value, axes []int64, keepDims bool) *Value {
	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	outShape := computeReduceShape(x.shape, axes, keepDims)

	return b.addOp("reduce_prod", map[string]*Value{
		"x":         x,
		"axes":      axesVal,
		"keep_dims": keepVal,
	}, b.genName("reduce_prod"), x.dtype, outShape)
}

// ArgMax returns indices of maximum values along an axis.
func (b *Builder) ArgMax(x *Value, axis int64, keepDims bool) *Value {
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	// ArgMax returns Int32 indices, not the input dtype
	outShape := computeReduceShape(x.shape, []int64{axis}, keepDims)

	return b.addOp("reduce_argmax", map[string]*Value{
		"x":         x,
		"axis":      axisVal,
		"keep_dims": keepVal,
	}, b.genName("argmax"), Int32, outShape)
}

// ArgMin returns indices of minimum values along an axis.
func (b *Builder) ArgMin(x *Value, axis int64, keepDims bool) *Value {
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})
	keepVal := b.Const(b.genName("keep"), Bool, []int64{}, []bool{keepDims})

	// ArgMin returns Int32 indices, not the input dtype
	outShape := computeReduceShape(x.shape, []int64{axis}, keepDims)

	return b.addOp("reduce_argmin", map[string]*Value{
		"x":         x,
		"axis":      axisVal,
		"keep_dims": keepVal,
	}, b.genName("argmin"), Int32, outShape)
}

// Equal performs element-wise equality comparison: z = (x == y).
// Returns Bool dtype.
func (b *Builder) Equal(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("equal", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("equal"), Bool, outShape)
}

// NotEqual performs element-wise inequality comparison: z = (x != y).
// Returns Bool dtype.
func (b *Builder) NotEqual(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("not_equal", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("not_equal"), Bool, outShape)
}

// Less performs element-wise less-than comparison: z = (x < y).
// Returns Bool dtype.
func (b *Builder) Less(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("less", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("less"), Bool, outShape)
}

// LessEqual performs element-wise less-than-or-equal comparison: z = (x <= y).
// Returns Bool dtype.
func (b *Builder) LessEqual(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("less_equal", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("less_equal"), Bool, outShape)
}

// Greater performs element-wise greater-than comparison: z = (x > y).
// Returns Bool dtype.
func (b *Builder) Greater(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("greater", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("greater"), Bool, outShape)
}

// GreaterEqual performs element-wise greater-than-or-equal comparison: z = (x >= y).
// Returns Bool dtype.
func (b *Builder) GreaterEqual(x, y *Value) *Value {
	outShape := broadcastShape(x.shape, y.shape)
	return b.addOp("greater_equal", map[string]*Value{
		"x": x,
		"y": y,
	}, b.genName("greater_equal"), Bool, outShape)
}

// Select performs element-wise selection based on a condition.
// Returns a where cond is true, b where cond is false.
// cond must have Bool dtype, a and b must have matching dtypes.
func (b *Builder) Select(cond, a, bVal *Value) *Value {
	outShape := broadcastShape(a.shape, bVal.shape)
	return b.addOp("select", map[string]*Value{
		"cond": cond,
		"a":    a,
		"b":    bVal,
	}, b.genName("select"), a.dtype, outShape)
}

// Squeeze removes dimensions of size 1 from the tensor shape.
// If axes is empty or nil, all dimensions of size 1 are removed.
func (b *Builder) Squeeze(x *Value, axes []int64) *Value {
	// Compute output shape by removing specified axes
	outShape := make([]int64, 0)

	if len(axes) == 0 {
		// Squeeze all size-1 dimensions
		for _, dim := range x.shape {
			if dim != 1 {
				outShape = append(outShape, dim)
			}
		}
	} else {
		// Build set of axes to squeeze
		axisSet := make(map[int64]bool)
		for _, a := range axes {
			if a < 0 {
				a = int64(len(x.shape)) + a
			}
			axisSet[a] = true
		}

		// Remove specified axes
		for i, dim := range x.shape {
			if !axisSet[int64(i)] {
				outShape = append(outShape, dim)
			}
		}
	}

	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))

	return b.addOp("squeeze", map[string]*Value{
		"x":    x,
		"axes": axesVal,
	}, b.genName("squeeze"), x.dtype, outShape)
}

// ExpandDims adds dimensions of size 1 at specified axes.
func (b *Builder) ExpandDims(x *Value, axes []int64) *Value {
	// Compute output shape by inserting size-1 dimensions
	outRank := len(x.shape) + len(axes)
	outShape := make([]int64, outRank)

	// Normalize and build set of axes where we insert size-1 dims
	normalizedAxes := make(map[int64]bool)
	for _, a := range axes {
		if a < 0 {
			a = int64(outRank) + a
		}
		normalizedAxes[a] = true
	}

	// Build output shape
	srcIdx := 0
	for i := 0; i < outRank; i++ {
		if normalizedAxes[int64(i)] {
			outShape[i] = 1
		} else {
			outShape[i] = x.shape[srcIdx]
			srcIdx++
		}
	}

	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))

	return b.addOp("expand_dims", map[string]*Value{
		"x":    x,
		"axes": axesVal,
	}, b.genName("expand_dims"), x.dtype, outShape)
}

// SliceByIndex extracts a sub-tensor using start/end indices along each axis.
// begin: starting indices for each dimension (inclusive)
// end: ending indices for each dimension (exclusive)
// strides: step size for each dimension (nil or empty defaults to 1)
func (b *Builder) SliceByIndex(x *Value, begin, end, strides []int64) *Value {
	// Handle nil or empty strides (default to 1)
	if len(strides) == 0 {
		strides = make([]int64, len(begin))
		for i := range strides {
			strides[i] = 1
		}
	}

	// Compute output shape
	outShape := make([]int64, len(x.shape))
	for i := range outShape {
		start := begin[i]
		stop := end[i]
		stride := strides[i]
		if stride == 0 {
			stride = 1
		}
		// Handle negative indices
		if start < 0 {
			start = x.shape[i] + start
		}
		if stop < 0 {
			stop = x.shape[i] + stop
		}
		// Compute output dimension size
		outShape[i] = (stop - start + stride - 1) / stride
	}

	beginVal := b.Const(b.genName("begin"), Int32, []int64{int64(len(begin))}, toInt32Slice(begin))
	endVal := b.Const(b.genName("end"), Int32, []int64{int64(len(end))}, toInt32Slice(end))
	stridesVal := b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))

	return b.addOp("slice_by_index", map[string]*Value{
		"x":      x,
		"begin":  beginVal,
		"end":    endVal,
		"stride": stridesVal,
	}, b.genName("slice"), x.dtype, outShape)
}

// Gather gathers values from x using indices along a specified axis.
// Output shape: x.shape[:axis] + indices.shape + x.shape[axis+1:]
func (b *Builder) Gather(x *Value, indices *Value, axis int64) *Value {
	// Handle negative axis
	if axis < 0 {
		axis = int64(len(x.shape)) + axis
	}

	// Compute output shape:
	// Replace x.shape[axis] with indices.shape
	outShape := make([]int64, 0, len(x.shape)-1+len(indices.shape))
	outShape = append(outShape, x.shape[:axis]...)
	outShape = append(outShape, indices.shape...)
	outShape = append(outShape, x.shape[axis+1:]...)

	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})

	return b.addOp("gather", map[string]*Value{
		"x":       x,
		"indices": indices,
		"axis":    axisVal,
	}, b.genName("gather"), x.dtype, outShape)
}

// BatchNorm applies batch normalization to the input tensor.
// x: Input tensor with shape [N, C, *D] where N is batch size, C is channels, *D are spatial dimensions (rank 3-5).
// mean: Channel-wise mean with shape [C].
// variance: Channel-wise variance with shape [C].
// gamma: Optional scale parameter with shape [C]. If nil, defaults to all ones.
// beta: Optional shift parameter with shape [C]. If nil, defaults to all zeros.
// epsilon: Small constant added to variance for numerical stability (typically 1e-5).
// Output shape is same as input x.
func (b *Builder) BatchNorm(x, mean, variance, gamma, beta *Value, epsilon float32) *Value {
	epsilonVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

	inputs := map[string]*Value{
		"x":        x,
		"mean":     mean,
		"variance": variance,
		"epsilon":  epsilonVal,
	}

	if gamma != nil {
		inputs["gamma"] = gamma
	}
	if beta != nil {
		inputs["beta"] = beta
	}

	return b.addOp("batch_norm", inputs, b.genName("batch_norm"), x.dtype, x.shape)
}

// LayerNorm applies layer normalization to the input tensor.
// x: Input tensor of any shape.
// gamma: Optional scale parameter with shape matching x.shape[axes]. If nil, defaults to all ones.
// beta: Optional shift parameter with shape matching x.shape[axes]. If nil, defaults to all zeros.
// axes: Dimensions along which to perform normalization. If nil or empty, normalizes over all axes.
// epsilon: Small constant added to variance for numerical stability (typically 1e-5).
// Output shape is same as input x.
func (b *Builder) LayerNorm(x, gamma, beta *Value, axes []int64, epsilon float32) *Value {
	epsilonVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

	inputs := map[string]*Value{
		"x":       x,
		"epsilon": epsilonVal,
	}

	if len(axes) > 0 {
		axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))
		inputs["axes"] = axesVal
	}
	if gamma != nil {
		inputs["gamma"] = gamma
	}
	if beta != nil {
		inputs["beta"] = beta
	}

	return b.addOp("layer_norm", inputs, b.genName("layer_norm"), x.dtype, x.shape)
}

// InstanceNorm applies instance normalization to the input tensor.
// x: Input tensor with shape [N, C, *D] where N is batch size, C is channels, *D are spatial dimensions (rank 3-4).
// gamma: Optional scale parameter with shape [C]. If nil, defaults to all ones.
// beta: Optional shift parameter with shape [C]. If nil, defaults to all zeros.
// epsilon: Small constant added to variance for numerical stability (typically 1e-5).
// Output shape is same as input x.
func (b *Builder) InstanceNorm(x, gamma, beta *Value, epsilon float32) *Value {
	epsilonVal := b.Const(b.genName("epsilon"), Float32, []int64{}, []float32{epsilon})

	inputs := map[string]*Value{
		"x":       x,
		"epsilon": epsilonVal,
	}

	if gamma != nil {
		inputs["gamma"] = gamma
	}
	if beta != nil {
		inputs["beta"] = beta
	}

	return b.addOp("instance_norm", inputs, b.genName("instance_norm"), x.dtype, x.shape)
}

// MaxPool applies max pooling operation.
// x: input tensor with shape [N, C, H, W] for 2D pooling
// kernelSize: size of pooling window for each spatial dimension
// strides: stride for each spatial dimension
// padType: padding type (ConvPadValid, ConvPadSame, or ConvPadCustom)
// padBefore, padAfter: custom padding (only used if padType == ConvPadCustom)
func (b *Builder) MaxPool(x *Value, kernelSize, strides []int64, padType ConvPadType, padBefore, padAfter []int64) *Value {
	// Compute output shape
	// For 2D: input [N, C, H, W] -> output [N, C, H_out, W_out]
	outShape := make([]int64, len(x.shape))
	copy(outShape[:2], x.shape[:2]) // Copy N, C dimensions

	// Compute spatial dimensions based on padding type
	for i := 0; i < len(kernelSize); i++ {
		spatialIdx := 2 + i
		inputSize := x.shape[spatialIdx]
		kernelSz := kernelSize[i]
		stride := strides[i]

		var padTotal int64
		switch padType {
		case ConvPadValid:
			padTotal = 0
		case ConvPadSame:
			// Output size equals input size when stride=1
			padTotal = (kernelSz - 1)
		case ConvPadCustom:
			padTotal = padBefore[i] + padAfter[i]
		}

		outShape[spatialIdx] = (inputSize + padTotal - kernelSz) / stride + 1
	}

	kernelVal := b.Const(b.genName("kernel_sizes"), Int32, []int64{int64(len(kernelSize))}, toInt32Slice(kernelSize))
	stridesVal := b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))

	inputs := map[string]*Value{
		"x":            x,
		"kernel_sizes": kernelVal,
		"strides":      stridesVal,
	}

	// Add padding parameters based on type
	switch padType {
	case ConvPadValid:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "valid")
		inputs["pad_type"] = padTypeVal
	case ConvPadSame:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "same")
		inputs["pad_type"] = padTypeVal
	case ConvPadCustom:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "custom")
		inputs["pad_type"] = padTypeVal
		// CoreML MIL uses a single "pad" parameter with format [before_0, after_0, before_1, after_1, ...]
		padSpec := make([]int32, 2*len(padBefore))
		for i := range padBefore {
			padSpec[2*i] = int32(padBefore[i])
			padSpec[2*i+1] = int32(padAfter[i])
		}
		padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
		inputs["pad"] = padVal
	}

	return b.addOp("max_pool", inputs, b.genName("max_pool"), x.dtype, outShape)
}

// AvgPool applies average pooling operation.
// x: input tensor with shape [N, C, H, W] for 2D pooling
// kernelSize: size of pooling window for each spatial dimension
// strides: stride for each spatial dimension
// padType: padding type (ConvPadValid, ConvPadSame, or ConvPadCustom)
// padBefore, padAfter: custom padding (only used if padType == ConvPadCustom)
// excludePaddingFromAverage: if true, exclude padding values from average calculation
func (b *Builder) AvgPool(x *Value, kernelSize, strides []int64, padType ConvPadType, padBefore, padAfter []int64, excludePaddingFromAverage bool) *Value {
	// Compute output shape (same as MaxPool)
	outShape := make([]int64, len(x.shape))
	copy(outShape[:2], x.shape[:2]) // Copy N, C dimensions

	for i := 0; i < len(kernelSize); i++ {
		spatialIdx := 2 + i
		inputSize := x.shape[spatialIdx]
		kernelSz := kernelSize[i]
		stride := strides[i]

		var padTotal int64
		switch padType {
		case ConvPadValid:
			padTotal = 0
		case ConvPadSame:
			padTotal = (kernelSz - 1)
		case ConvPadCustom:
			padTotal = padBefore[i] + padAfter[i]
		}

		outShape[spatialIdx] = (inputSize + padTotal - kernelSz) / stride + 1
	}

	kernelVal := b.Const(b.genName("kernel_sizes"), Int32, []int64{int64(len(kernelSize))}, toInt32Slice(kernelSize))
	stridesVal := b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))
	excludeVal := b.Const(b.genName("exclude_padding"), Bool, []int64{}, []bool{excludePaddingFromAverage})

	inputs := map[string]*Value{
		"x":                            x,
		"kernel_sizes":                 kernelVal,
		"strides":                      stridesVal,
		"exclude_padding_from_average": excludeVal,
	}

	// Add padding parameters based on type
	switch padType {
	case ConvPadValid:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "valid")
		inputs["pad_type"] = padTypeVal
	case ConvPadSame:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "same")
		inputs["pad_type"] = padTypeVal
	case ConvPadCustom:
		padTypeVal := b.Const(b.genName("pad_type"), String, []int64{}, "custom")
		inputs["pad_type"] = padTypeVal
		// Interleave padding values
		padSpec := make([]int32, 2*len(padBefore))
		for i := range padBefore {
			padSpec[2*i] = int32(padBefore[i])
			padSpec[2*i+1] = int32(padAfter[i])
		}
		padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)
		inputs["pad"] = padVal
	}

	return b.addOp("avg_pool", inputs, b.genName("avg_pool"), x.dtype, outShape)
}

// GlobalAvgPool2D applies global average pooling over spatial dimensions (H, W).
// For NCHW input, reduces over dimensions 2 and 3.
// Output has shape [N, C, 1, 1] with keepDims=true.
func (b *Builder) GlobalAvgPool2D(x *Value) *Value {
	// Reduce over H, W dimensions (axes 2, 3 for NCHW)
	return b.ReduceMean(x, []int64{2, 3}, true)
}

// GlobalMaxPool2D applies global max pooling over spatial dimensions (H, W).
// For NCHW input, reduces over dimensions 2 and 3.
// Output has shape [N, C, 1, 1] with keepDims=true.
func (b *Builder) GlobalMaxPool2D(x *Value) *Value {
	// Reduce over H, W dimensions (axes 2, 3 for NCHW)
	return b.ReduceMax(x, []int64{2, 3}, true)
}

// Tile repeats a tensor along each axis by the specified repetition factors.
// reps: number of repetitions for each dimension
// Output shape: [x.shape[i] * reps[i] for i in range(rank)]
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

// PadMode specifies the padding mode for Pad operation.
type PadMode int

const (
	// PadConstant fills padded values with a constant value.
	PadConstant PadMode = iota
	// PadReflect reflects values at the boundaries (mirroring without repeating edge values).
	PadReflect
	// PadReplicate replicates edge values.
	PadReplicate
)

// Pad adds padding to a tensor.
// padBefore: number of values to pad before each dimension
// padAfter: number of values to pad after each dimension
// mode: padding mode (constant, reflect, or replicate)
// constantValue: value to use for constant padding (ignored for other modes)
// Output shape: [x.shape[i] + padBefore[i] + padAfter[i] for i in range(rank)]
func (b *Builder) Pad(x *Value, padBefore, padAfter []int64, mode PadMode, constantValue float32) *Value {
	// Compute output shape
	outShape := make([]int64, len(x.shape))
	for i := range outShape {
		outShape[i] = x.shape[i] + padBefore[i] + padAfter[i]
	}

	// Create pad specification: [before_0, after_0, before_1, after_1, ...]
	padSpec := make([]int32, 2*len(padBefore))
	for i := range padBefore {
		padSpec[2*i] = int32(padBefore[i])
		padSpec[2*i+1] = int32(padAfter[i])
	}
	padVal := b.Const(b.genName("pad"), Int32, []int64{int64(len(padSpec))}, padSpec)

	// Convert mode to string
	var modeStr string
	switch mode {
	case PadConstant:
		modeStr = "constant"
	case PadReflect:
		modeStr = "reflect"
	case PadReplicate:
		modeStr = "replicate"
	default:
		modeStr = "constant"
	}
	modeVal := b.Const(b.genName("mode"), String, []int64{}, modeStr)

	// Constant value (only used for constant mode)
	constVal := b.Const(b.genName("constant_val"), Float32, []int64{}, []float32{constantValue})

	return b.addOp("pad", map[string]*Value{
		"x":            x,
		"pad":          padVal,
		"mode":         modeVal,
		"constant_val": constVal,
	}, b.genName("pad"), x.dtype, outShape)
}

// Reverse reverses a tensor along specified axes.
// axes: axes along which to reverse (empty or nil reverses all axes)
// Output shape: same as input shape
func (b *Builder) Reverse(x *Value, axes []int64) *Value {
	// If axes is empty, reverse along all axes
	if len(axes) == 0 {
		axes = make([]int64, len(x.shape))
		for i := range axes {
			axes[i] = int64(i)
		}
	}

	axesVal := b.Const(b.genName("axes"), Int32, []int64{int64(len(axes))}, toInt32Slice(axes))

	return b.addOp("reverse", map[string]*Value{
		"x":    x,
		"axes": axesVal,
	}, b.genName("reverse"), x.dtype, x.shape)
}

// Conv performs 2D convolution on input tensor x with filter weights.
// x: Input tensor in NCHW format [batch, channels_in, height, width]
// weight: Filter tensor [channels_out, channels_in/groups, kernel_height, kernel_width]
// strides: Stride for each spatial dimension [stride_h, stride_w]
// dilations: Dilation for each spatial dimension [dilation_h, dilation_w]
// padType: Padding type (ConvPadValid, ConvPadSame, or ConvPadCustom)
// padBefore: Padding before each spatial dimension [pad_h_before, pad_w_before] (used only if padType is ConvPadCustom)
// padAfter: Padding after each spatial dimension [pad_h_after, pad_w_after] (used only if padType is ConvPadCustom)
// groups: Number of groups for grouped convolution (1 for standard convolution)
func (b *Builder) Conv(x, weight *Value, strides, dilations []int64, padType ConvPadType, padBefore, padAfter []int64, groups int64) *Value {
	// Input shape: [N, C_in, H, W]
	// Weight shape: [C_out, C_in/groups, kH, kW]
	N := x.shape[0]
	Cout := weight.shape[0]
	kH := weight.shape[2]
	kW := weight.shape[3]
	inH := x.shape[2]
	inW := x.shape[3]

	// Default strides and dilations if not provided
	if len(strides) == 0 {
		strides = []int64{1, 1}
	}
	if len(dilations) == 0 {
		dilations = []int64{1, 1}
	}

	// Compute output spatial dimensions based on padding type
	var outH, outW int64
	var padTypeStr string

	switch padType {
	case ConvPadValid:
		padTypeStr = "valid"
		// Output size with no padding
		outH = (inH - dilations[0]*(kH-1) - 1) / strides[0] + 1
		outW = (inW - dilations[1]*(kW-1) - 1) / strides[1] + 1

	case ConvPadSame:
		padTypeStr = "same"
		// Output size preserves input dimensions (accounting for stride)
		outH = (inH + strides[0] - 1) / strides[0]
		outW = (inW + strides[1] - 1) / strides[1]

	case ConvPadCustom:
		padTypeStr = "custom"
		// Compute output size with custom padding
		if len(padBefore) == 0 {
			padBefore = []int64{0, 0}
		}
		if len(padAfter) == 0 {
			padAfter = []int64{0, 0}
		}
		paddedH := inH + padBefore[0] + padAfter[0]
		paddedW := inW + padBefore[1] + padAfter[1]
		outH = (paddedH - dilations[0]*(kH-1) - 1) / strides[0] + 1
		outW = (paddedW - dilations[1]*(kW-1) - 1) / strides[1] + 1
	}

	outShape := []int64{N, Cout, outH, outW}

	// Build operation inputs
	inputs := map[string]*Value{
		"x":      x,
		"weight": weight,
	}

	// Add parameters
	inputs["strides"] = b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))
	inputs["dilations"] = b.Const(b.genName("dilations"), Int32, []int64{int64(len(dilations))}, toInt32Slice(dilations))
	inputs["groups"] = b.Const(b.genName("groups"), Int32, []int64{}, []int32{int32(groups)})
	inputs["pad_type"] = b.Const(b.genName("pad_type"), String, []int64{}, padTypeStr)

	if padType == ConvPadCustom {
		// Flatten padding into [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
		pad := []int64{padBefore[0], padAfter[0], padBefore[1], padAfter[1]}
		inputs["pad"] = b.Const(b.genName("pad"), Int32, []int64{4}, toInt32Slice(pad))
	}

	return b.addOp("conv", inputs, b.genName("conv"), x.dtype, outShape)
}

// ConvTranspose performs 2D transposed convolution (also known as deconvolution).
// x: Input tensor in NCHW format [batch, channels_in, height, width]
// weight: Filter tensor [channels_in, channels_out/groups, kernel_height, kernel_width]
// strides: Stride for each spatial dimension [stride_h, stride_w]
// dilations: Dilation for each spatial dimension [dilation_h, dilation_w]
// padType: Padding type (ConvPadValid, ConvPadSame, or ConvPadCustom)
// padBefore: Padding before each spatial dimension [pad_h_before, pad_w_before] (used only if padType is ConvPadCustom)
// padAfter: Padding after each spatial dimension [pad_h_after, pad_w_after] (used only if padType is ConvPadCustom)
// outputPadding: Additional padding added to output [output_pad_h, output_pad_w]
// groups: Number of groups for grouped convolution (1 for standard convolution)
func (b *Builder) ConvTranspose(x, weight *Value, strides, dilations []int64, padType ConvPadType, padBefore, padAfter, outputPadding []int64, groups int64) *Value {
	// Input shape: [N, C_in, H, W]
	// Weight shape: [C_in, C_out/groups, kH, kW]
	N := x.shape[0]
	Cout := weight.shape[1] * groups
	kH := weight.shape[2]
	kW := weight.shape[3]
	inH := x.shape[2]
	inW := x.shape[3]

	// Default strides and dilations if not provided
	if len(strides) == 0 {
		strides = []int64{1, 1}
	}
	if len(dilations) == 0 {
		dilations = []int64{1, 1}
	}
	if len(outputPadding) == 0 {
		outputPadding = []int64{0, 0}
	}

	// Compute output spatial dimensions based on padding type
	var outH, outW int64
	var padTypeStr string

	switch padType {
	case ConvPadValid:
		padTypeStr = "valid"
		// Transposed convolution output size with no padding
		outH = (inH-1)*strides[0] + dilations[0]*(kH-1) + 1 + outputPadding[0]
		outW = (inW-1)*strides[1] + dilations[1]*(kW-1) + 1 + outputPadding[1]

	case ConvPadSame:
		padTypeStr = "same"
		// Output size preserves input dimensions (accounting for stride)
		outH = inH * strides[0]
		outW = inW * strides[1]

	case ConvPadCustom:
		padTypeStr = "custom"
		// Compute output size with custom padding
		if len(padBefore) == 0 {
			padBefore = []int64{0, 0}
		}
		if len(padAfter) == 0 {
			padAfter = []int64{0, 0}
		}
		outH = (inH-1)*strides[0] + dilations[0]*(kH-1) + 1 - padBefore[0] - padAfter[0] + outputPadding[0]
		outW = (inW-1)*strides[1] + dilations[1]*(kW-1) + 1 - padBefore[1] - padAfter[1] + outputPadding[1]
	}

	outShape := []int64{N, Cout, outH, outW}

	// Build operation inputs
	inputs := map[string]*Value{
		"x":      x,
		"weight": weight,
	}

	// Add parameters
	inputs["strides"] = b.Const(b.genName("strides"), Int32, []int64{int64(len(strides))}, toInt32Slice(strides))
	inputs["dilations"] = b.Const(b.genName("dilations"), Int32, []int64{int64(len(dilations))}, toInt32Slice(dilations))
	inputs["groups"] = b.Const(b.genName("groups"), Int32, []int64{}, []int32{int32(groups)})
	inputs["pad_type"] = b.Const(b.genName("pad_type"), String, []int64{}, padTypeStr)

	if padType == ConvPadCustom {
		// Flatten padding into [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
		pad := []int64{padBefore[0], padAfter[0], padBefore[1], padAfter[1]}
		inputs["pad"] = b.Const(b.genName("pad"), Int32, []int64{4}, toInt32Slice(pad))
	}

	if outputPadding[0] != 0 || outputPadding[1] != 0 {
		inputs["output_padding"] = b.Const(b.genName("output_padding"), Int32, []int64{int64(len(outputPadding))}, toInt32Slice(outputPadding))
	}

	return b.addOp("conv_transpose", inputs, b.genName("conv_transpose"), x.dtype, outShape)
}

// ConvWithBias performs 2D convolution with bias addition.
// This is a convenience function that combines Conv and bias addition.
// bias: Bias tensor [channels_out] to add to each output channel
// Other parameters are the same as Conv.
func (b *Builder) ConvWithBias(x, weight, bias *Value, strides, dilations []int64, padType ConvPadType, padBefore, padAfter []int64, groups int64) *Value {
	// Perform convolution
	conv := b.Conv(x, weight, strides, dilations, padType, padBefore, padAfter, groups)

	// Reshape bias for broadcasting: [C_out] -> [1, C_out, 1, 1]
	biasShape := []int64{1, bias.shape[0], 1, 1}
	biasReshaped := b.Reshape(bias, biasShape)

	// Add bias to convolution output
	return b.Add(conv, biasReshaped)
}

// Concat concatenates a list of tensors along a specified axis.
// values: List of tensors to concatenate. All must have the same shape except along the concat axis.
// axis: Axis along which to concatenate. Must be in range [-rank, rank).
// Output shape: same as input shapes, except dimension along axis is the sum of input dimensions.
func (b *Builder) Concat(values []*Value, axis int64) *Value {
	if len(values) == 0 {
		panic("concat requires at least one input tensor")
	}

	// Get first tensor's properties
	firstValue := values[0]
	dtype := firstValue.dtype
	rank := len(firstValue.shape)

	// Normalize negative axis
	if axis < 0 {
		axis = int64(rank) + axis
	}

	// Compute output shape: sum dimensions along concat axis
	outShape := make([]int64, rank)
	copy(outShape, firstValue.shape)

	// Sum the concat axis dimension across all inputs
	concatDim := int64(0)
	for _, v := range values {
		concatDim += v.shape[axis]
	}
	outShape[axis] = concatDim

	// Create axis constant
	axisVal := b.Const(b.genName("axis"), Int32, []int64{}, []int32{int32(axis)})

	// Use addOpWithListArg to handle the list of values
	return b.addOpWithListArg("concat",
		map[string]*Value{"axis": axisVal}, // scalar inputs
		map[string][]*Value{"values": values}, // list inputs
		b.genName("concat"),
		dtype,
		outShape)
}

// Helper functions

func toInt32Slice(s []int64) []int32 {
	result := make([]int32, len(s))
	for i, v := range s {
		result[i] = int32(v)
	}
	return result
}

func broadcastShape(a, b []int64) []int64 {
	maxLen := len(a)
	if len(b) > maxLen {
		maxLen = len(b)
	}

	result := make([]int64, maxLen)
	for i := 0; i < maxLen; i++ {
		ai := int64(1)
		bi := int64(1)

		if i < len(a) {
			ai = a[len(a)-1-i]
		}
		if i < len(b) {
			bi = b[len(b)-1-i]
		}

		if ai == 1 {
			result[maxLen-1-i] = bi
		} else if bi == 1 {
			result[maxLen-1-i] = ai
		} else if ai == bi {
			result[maxLen-1-i] = ai
		} else {
			// Incompatible shapes - return larger
			if ai > bi {
				result[maxLen-1-i] = ai
			} else {
				result[maxLen-1-i] = bi
			}
		}
	}
	return result
}

func computeReduceShape(shape []int64, axes []int64, keepDims bool) []int64 {
	axisSet := make(map[int64]bool)
	for _, a := range axes {
		if a < 0 {
			a = int64(len(shape)) + a
		}
		axisSet[a] = true
	}

	if keepDims {
		result := make([]int64, len(shape))
		for i, dim := range shape {
			if axisSet[int64(i)] {
				result[i] = 1
			} else {
				result[i] = dim
			}
		}
		return result
	}

	var result []int64
	for i, dim := range shape {
		if !axisSet[int64(i)] {
			result = append(result, dim)
		}
	}
	if len(result) == 0 {
		return []int64{} // Scalar
	}
	return result
}
