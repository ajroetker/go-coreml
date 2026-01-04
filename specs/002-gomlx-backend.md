# GoMLX CoreML Backend Implementation Plan

## Status

**Phases 1-2: Complete**
- Bridge package with cgo bindings to CoreML
- MIL program builder with fluent API
- Model serialization (.mlpackage format)
- Runtime compilation and execution
- Basic operations (Add, Sub, Mul, Div, MatMul, Relu, Sigmoid, etc.)

**Phases 3-5: This Document**

---

## Phase 3: GoMLX Backend Integration

### Overview

Implement GoMLX's backend interfaces using go-coreml. This creates the bridge between GoMLX computation graphs and CoreML execution.

### 3.1 Study GoMLX Backend Interfaces

**Files to Read**:
```
gomlx/backends/backends.go      - Backend interface
gomlx/backends/builder.go       - Builder interface
gomlx/backends/executable.go    - Executable interface
gomlx/backends/data.go          - DataInterface (buffers)
gomlx/backends/standard_ops.go  - StandardOps enum
gomlx/backends/simplego/        - Reference implementation (pure Go)
gomlx/backends/xla/             - Production reference (XLA)
```

**Key Questions to Answer**:
1. What methods does `Backend` interface require?
2. What is the lifecycle of `Builder` → `Executable`?
3. How does `DataInterface` handle buffer management?
4. What is `Node` and how does it relate to operations?
5. How does shape inference work?

### 3.2 Implement Backend Interface

**Location**: `github.com/gomlx/gomlx/backends/coreml/backend.go`

```go
type Backend struct {
    computeUnits bridge.ComputeUnits
    cacheDir     string
}

// Required methods (study interface to confirm):
func (b *Backend) Name() string
func (b *Backend) NewBuilder(name string) backends.Builder
func (b *Backend) NewBuffer(shape shapes.Shape) backends.Buffer
func (b *Backend) Platform() string
func (b *Backend) Close() error
```

**Tasks**:
- [ ] Read GoMLX Backend interface definition
- [ ] Implement all required methods
- [ ] Add compute unit configuration (ANE, GPU, CPU, All)
- [ ] Handle platform detection (macOS only)
- [ ] Register backend with GoMLX registry

### 3.3 Implement Builder Interface

**Location**: `github.com/gomlx/gomlx/backends/coreml/builder.go`

```go
type Builder struct {
    backend    *Backend
    milBuilder *model.Builder
    nodeMap    map[backends.NodeID]*Node
    nextNodeID backends.NodeID
}

type Node struct {
    id      backends.NodeID
    value   *model.Value
    shape   shapes.Shape
    dtype   dtypes.DType
}

// Key methods:
func (b *Builder) Parameter(name string, shape shapes.Shape) backends.Node
func (b *Builder) Constant(value interface{}, shape shapes.Shape) backends.Node
func (b *Builder) Op(op backends.StandardOp, inputs ...backends.Node) backends.Node
func (b *Builder) Compile(outputs []backends.Node) backends.Executable
```

**Tasks**:
- [ ] Study Builder interface requirements
- [ ] Implement Parameter (maps to model.Input)
- [ ] Implement Constant (maps to model.Const)
- [ ] Create operation dispatch table
- [ ] Implement shape inference helpers
- [ ] Implement dtype conversion (GoMLX dtypes → CoreML dtypes)

### 3.4 Implement Executable Interface

**Location**: `github.com/gomlx/gomlx/backends/coreml/executable.go`

```go
type Executable struct {
    backend     *Backend
    runtime     *runtime.Executable
    inputNames  []string
    outputNames []string
    inputShapes []shapes.Shape
    outputShapes []shapes.Shape
}

// Key methods:
func (e *Executable) Execute(inputs []backends.Buffer) ([]backends.Buffer, error)
func (e *Executable) Close() error
func (e *Executable) InputShapes() []shapes.Shape
func (e *Executable) OutputShapes() []shapes.Shape
```

**Tasks**:
- [ ] Study Executable interface requirements
- [ ] Implement Execute with buffer conversion
- [ ] Handle input/output shape validation
- [ ] Implement proper cleanup in Close()

### 3.5 Implement Buffer/DataInterface

**Location**: `github.com/gomlx/gomlx/backends/coreml/buffer.go`

```go
type Buffer struct {
    data  []byte
    shape shapes.Shape
    dtype dtypes.DType
}

// Methods:
func (buf *Buffer) Shape() shapes.Shape
func (buf *Buffer) DType() dtypes.DType
func (buf *Buffer) Bytes() []byte
func (buf *Buffer) CopyFrom(data interface{})
func (buf *Buffer) CopyTo(dst interface{})
```

**Tasks**:
- [ ] Study DataInterface requirements
- [ ] Implement buffer creation and management
- [ ] Handle type conversions (Go types ↔ CoreML types)
- [ ] Implement efficient memory copying

### 3.6 Operation Mapping Layer

**Location**: `github.com/gomlx/gomlx/backends/coreml/ops.go`

Create dispatch table mapping GoMLX StandardOps to MIL operations:

```go
type opHandler func(b *Builder, inputs []*Node) (*Node, error)

var opTable = map[backends.StandardOp]opHandler{
    backends.OpAdd:     handleAdd,
    backends.OpSub:     handleSub,
    backends.OpMul:     handleMul,
    backends.OpDiv:     handleDiv,
    backends.OpMatMul:  handleMatMul,
    backends.OpRelu:    handleRelu,
    // ... etc
}

func (b *Builder) dispatchOp(op backends.StandardOp, inputs []*Node) (*Node, error) {
    handler, ok := opTable[op]
    if !ok {
        return nil, fmt.Errorf("unsupported operation: %v", op)
    }
    return handler(b, inputs)
}
```

**Tasks**:
- [ ] List all StandardOps from GoMLX
- [ ] Create handler for each supported op
- [ ] Implement shape inference per operation
- [ ] Handle broadcasting rules
- [ ] Return clear errors for unsupported ops

### 3.7 Backend Registration

Use build tags to conditionally compile the backend only on macOS.

**Location**: `github.com/gomlx/gomlx/backends/coreml/register_darwin.go`

```go
//go:build darwin

package coreml

func init() {
    backends.Register("coreml", func() backends.Backend {
        return New()
    })
}
```

**Location**: `github.com/gomlx/gomlx/backends/coreml/register_other.go`

```go
//go:build !darwin

package coreml

// No-op on non-macOS platforms - backend not registered
```

This approach:
- Keeps the package importable on all platforms
- Avoids runtime checks
- Lets the compiler exclude CoreML code entirely on non-Apple platforms
- Prevents linker errors from missing Objective-C frameworks

### 3.8 Integration Testing

Create tests that use GoMLX API with CoreML backend:

```go
func TestGoMLXIntegration(t *testing.T) {
    backend := coreml.New()

    // Use GoMLX graph API
    g := graph.New(backend)
    x := g.Parameter("x", shapes.Make(dtypes.Float32, 2, 3))
    y := g.Relu(x)

    exec := g.Compile(y)
    defer exec.Close()

    input := []float32{-1, 2, -3, 4, -5, 6}
    output, err := exec.Execute(input)
    // ... verify output
}
```

**Tasks**:
- [ ] Test basic operations through GoMLX API
- [ ] Test multi-operation graphs
- [ ] Test parameter passing
- [ ] Test shape inference
- [ ] Verify numerical correctness vs simplego backend

---

## Phase 4: Operation Completeness

### 4.1 Core Operations (Priority 1)

These operations are essential for most models:

| GoMLX Op | MIL Op | Status | Notes |
|----------|--------|--------|-------|
| Add | add | Done | go-coreml |
| Sub | sub | Done | go-coreml |
| Mul | mul | Done | go-coreml |
| Div | real_div | Done | go-coreml |
| MatMul | matmul | Done | go-coreml |
| Relu | relu | Done | go-coreml |
| Sigmoid | sigmoid | Done | go-coreml |
| Tanh | tanh | Done | go-coreml |
| Exp | exp | Done | go-coreml |
| Log | log | Done | go-coreml |
| Sqrt | sqrt | Done | go-coreml |
| Neg | neg | Done | go-coreml |
| Abs | abs | Done | go-coreml |
| Softmax | softmax | Done | go-coreml |
| Reshape | reshape | Done | go-coreml |
| Transpose | transpose | Done | go-coreml |
| ReduceSum | reduce_sum | Done | go-coreml |
| ReduceMean | reduce_mean | Done | go-coreml |
| ReduceMax | reduce_max | Done | go-coreml |

### 4.2 Convolution Operations (Priority 2)

Essential for vision models:

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Conv2D | conv | Need to handle padding, strides, dilation |
| ConvTranspose2D | conv_transpose | Deconvolution |
| MaxPool2D | max_pool | Need to handle padding, strides |
| AvgPool2D | avg_pool | Need to handle padding, strides |

**Tasks**:
- [ ] Implement Conv2D with all parameters
- [ ] Handle NHWC vs NCHW format conversion
- [ ] Implement pooling operations
- [ ] Test with simple CNN architectures

### 4.3 Normalization Operations (Priority 2)

Essential for deep networks:

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| BatchNorm | batch_norm | Need running mean/var |
| LayerNorm | layer_norm | |
| InstanceNorm | instance_norm | |

**Tasks**:
- [ ] Implement BatchNorm with training/inference modes
- [ ] Implement LayerNorm
- [ ] Test normalization numerical accuracy

### 4.4 Tensor Manipulation (Priority 2)

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Concat | concat | Along specified axis |
| Split | split | |
| Slice | slice_by_index | |
| Gather | gather | |
| Scatter | scatter | May need decomposition |
| Stack | stack | |
| Squeeze | squeeze | |
| ExpandDims | expand_dims | |
| Tile | tile | |
| Pad | pad | |

**Tasks**:
- [ ] Implement each operation
- [ ] Handle axis/dimension parameters correctly
- [ ] Test edge cases (empty tensors, single elements)

### 4.5 Comparison Operations (Priority 3)

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Equal | equal | |
| NotEqual | not_equal | |
| Less | less | |
| LessEqual | less_equal | |
| Greater | greater | |
| GreaterEqual | greater_equal | |
| Where | select | Conditional selection |

### 4.6 Additional Math Operations (Priority 3)

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Pow | pow | |
| Sin, Cos, Tan | sin, cos, tan | |
| Floor, Ceil | floor, ceil | |
| Clip | clip | |
| Gelu | gelu | |
| Erf | erf | |

### 4.7 Attention Operations (Priority 3)

For transformer models:

| GoMLX Op | MIL Op | Notes |
|----------|--------|-------|
| Einsum | Decompose | Break into matmul/transpose |
| ScaledDotProductAttention | scaled_dot_product_attention | CoreML 7+ |

**Tasks**:
- [ ] Implement einsum decomposition
- [ ] Use native attention op when available
- [ ] Test with transformer architectures

### 4.8 Composite Operations

Some GoMLX operations need multiple MIL ops:

```go
// Example: Einsum decomposition
func handleEinsum(b *Builder, equation string, inputs []*Node) (*Node, error) {
    // Parse einsum equation
    // Decompose into sequence of:
    // - transpose
    // - reshape
    // - matmul
    // - reduce
}
```

**Tasks**:
- [ ] Identify operations needing decomposition
- [ ] Implement decomposition strategies
- [ ] Verify numerical equivalence

### 4.9 Test Suite

Create comprehensive tests for each operation:

```go
func TestOp_Add(t *testing.T) {
    testCases := []struct{
        name string
        a, b []float32
        aShape, bShape shapes.Shape
        want []float32
    }{
        {"simple", []float32{1,2}, []float32{3,4}, ...},
        {"broadcast", ...},
        {"scalar", ...},
    }
    // Test each case
}
```

**Tasks**:
- [ ] Create test cases for each operation
- [ ] Test broadcasting behavior
- [ ] Test edge cases (empty, scalar, large)
- [ ] Compare results vs simplego backend
- [ ] Add fuzzing tests for numerical stability

---

## Phase 5: Optimization & Benchmarking

### 5.1 Memory Management

**Buffer Pool**:
```go
type BufferPool struct {
    pools map[int]*sync.Pool // keyed by size
}

func (p *BufferPool) Get(size int) *Buffer
func (p *BufferPool) Put(buf *Buffer)
```

**Tasks**:
- [ ] Implement buffer pooling
- [ ] Reduce allocations in hot paths
- [ ] Profile memory usage
- [ ] Add memory usage metrics

### 5.2 Model Caching

Cache compiled CoreML models to avoid recompilation:

```go
type ModelCache struct {
    dir      string
    mu       sync.RWMutex
    entries  map[string]*cacheEntry
}

type cacheEntry struct {
    path      string
    hash      string
    lastUsed  time.Time
    executable *runtime.Executable
}

func (c *ModelCache) Get(programHash string) (*runtime.Executable, bool)
func (c *ModelCache) Put(programHash string, exec *runtime.Executable)
```

**Tasks**:
- [ ] Compute stable hash of MIL programs
- [ ] Cache compiled .mlmodelc directories
- [ ] Implement LRU eviction
- [ ] Add cache statistics

### 5.3 Compute Unit Selection

```go
type ComputeConfig struct {
    Units     ComputeUnits // All, CPUOnly, CPUAndGPU, CPUAndANE
    AllowANE  bool         // Allow Neural Engine
    AllowGPU  bool         // Allow Metal GPU
    Fallback  bool         // Allow fallback to CPU
}

func (b *Backend) WithComputeConfig(cfg ComputeConfig) *Backend
```

**Tasks**:
- [ ] Expose compute unit configuration
- [ ] Test each configuration
- [ ] Document performance characteristics
- [ ] Auto-select based on model characteristics

### 5.4 Benchmarking Infrastructure

Create comprehensive benchmarks:

```go
func BenchmarkMatMul(b *testing.B) {
    sizes := []int{64, 128, 256, 512, 1024, 2048}
    backends := []string{"coreml", "xla", "simplego"}

    for _, size := range sizes {
        for _, backend := range backends {
            b.Run(fmt.Sprintf("%s/%d", backend, size), func(b *testing.B) {
                // Benchmark matmul of size x size
            })
        }
    }
}
```

**Benchmark Categories**:
1. **Micro-benchmarks**: Individual operations
2. **Layer benchmarks**: Conv, Attention, FFN blocks
3. **Model benchmarks**: Full model inference

### 5.5 Performance Targets

| Workload | XLA CPU | CoreML Target | Notes |
|----------|---------|---------------|-------|
| MatMul 1024x1024 | X ms | < X ms | ANE should excel |
| Conv2D 224x224 | X ms | < X ms | Vision workload |
| Transformer layer | X ms | < X ms | Attention + FFN |
| BERT inference | X ms | < X ms | Full model |
| ResNet-50 | X ms | < X ms | Vision model |

**Tasks**:
- [ ] Establish XLA CPU baselines on Apple Silicon
- [ ] Measure CoreML performance per workload
- [ ] Identify and fix performance regressions
- [ ] Document performance vs XLA

### 5.6 Profiling Tools

```go
type Profiler struct {
    enabled bool
    traces  []TraceEvent
}

type TraceEvent struct {
    Name      string
    Start     time.Time
    Duration  time.Duration
    Op        string
    Shape     shapes.Shape
}

func (p *Profiler) Start(name string) func()
func (p *Profiler) Report() string
```

**Tasks**:
- [ ] Add timing instrumentation
- [ ] Integrate with Instruments.app
- [ ] Create performance reports
- [ ] Add ANE/GPU utilization tracking

### 5.7 Optimization Techniques

1. **Operation Fusion**: Combine sequential operations
   ```go
   // Fuse: MatMul + Add + Relu → fused_matmul_add_relu
   func fuseOperations(ops []*Operation) []*Operation
   ```

2. **Layout Optimization**: Choose optimal tensor layout
   ```go
   // Prefer NHWC for ANE, NCHW for GPU
   func optimizeLayout(model *Model, target ComputeUnits) *Model
   ```

3. **Constant Folding**: Pre-compute constant expressions
   ```go
   func foldConstants(ops []*Operation) []*Operation
   ```

**Tasks**:
- [ ] Implement operation fusion for common patterns
- [ ] Add layout optimization pass
- [ ] Implement constant folding
- [ ] Measure impact of each optimization

### 5.8 CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks
on: [push, pull_request]
jobs:
  benchmark:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmarks
        run: go test -bench=. -benchmem ./...
      - name: Compare to baseline
        run: ./scripts/compare-benchmarks.sh
```

**Tasks**:
- [ ] Set up macOS CI runners
- [ ] Automate benchmark comparison
- [ ] Track performance over time
- [ ] Alert on regressions

---

## Deliverables Summary

### Phase 3 Deliverables
- [ ] Working GoMLX CoreML backend
- [ ] Backend registration with GoMLX
- [ ] Basic operations through GoMLX API
- [ ] Integration tests passing

### Phase 4 Deliverables
- [ ] 80%+ GoMLX operations supported
- [ ] Convolution operations working
- [ ] Normalization operations working
- [ ] Comprehensive test suite

### Phase 5 Deliverables
- [ ] Performance exceeds XLA CPU on Apple Silicon
- [ ] Memory-efficient buffer management
- [ ] Model caching implemented
- [ ] Benchmark suite and reports
- [ ] Documentation complete

---

## Dependencies

**Go Dependencies**:
- `github.com/gomlx/gomlx` - GoMLX framework
- `github.com/gomlx/go-coreml` - This package

**System Requirements**:
- macOS 12+ (for CoreML 5+)
- Xcode Command Line Tools
- Apple Silicon or Intel Mac

---

## Timeline Estimates

| Phase | Scope | Complexity |
|-------|-------|------------|
| Phase 3 | Backend integration | Medium - requires understanding GoMLX interfaces |
| Phase 4 | Operation completeness | Medium-High - many operations, edge cases |
| Phase 5 | Optimization | High - performance tuning is iterative |

---

## Open Questions

1. **Dynamic Shapes**: How does GoMLX handle dynamic shapes? CoreML requires shape ranges.
2. **Gradient Support**: If training is ever needed, CoreML has limited gradient support.
3. **Multi-device**: Should we support running on multiple compute units simultaneously?
4. **iOS/tvOS**: Should the backend support non-macOS Apple platforms?
