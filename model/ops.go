package model

// This file contains MIL operation builders.
// MIL operations are documented at:
// https://apple.github.io/coremltools/docs-guides/source/ops-reference.html

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
