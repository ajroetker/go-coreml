# go-coreml

Go bindings to Apple's CoreML framework for high-performance machine learning inference on Apple Silicon.

## Overview

go-coreml provides Go bindings to CoreML, enabling:

- Running ML models on Apple's Neural Engine (ANE)
- Metal GPU acceleration
- Programmatic model construction using MIL (Machine Learning Intermediate Language)
- Integration with [GoMLX](https://github.com/gomlx/gomlx) as a backend (planned)

## Status

**Alpha** - Core functionality is implemented but the API may change.

### Implemented

- [x] Low-level bridge to CoreML (tensor creation, model loading, inference)
- [x] Protobuf types generated from CoreML MIL.proto
- [x] MIL program builder with common operations (add, mul, matmul, relu, etc.)
- [x] Model serialization to .mlpackage format
- [x] Runtime for compiling and executing MIL programs

### Planned

- [ ] GoMLX backend integration
- [ ] More MIL operations (conv2d, pooling, etc.)
- [ ] Weight blob support for large models
- [ ] Performance benchmarks

## Requirements

- macOS 12.0+ (Monterey or later)
- Xcode (full installation for coremlcompiler)
- Go 1.21+

## Installation

```bash
go get github.com/gomlx/go-coreml
```

## Usage

### Building a MIL Program

```go
package main

import (
    "fmt"
    "github.com/gomlx/go-coreml/model"
    "github.com/gomlx/go-coreml/runtime"
)

func main() {
    // Build a simple model: y = relu(x)
    b := model.NewBuilder("main")
    x := b.Input("x", model.Float32, 2, 3)
    y := b.Relu(x)
    b.Output("y", y)

    // Compile and load
    rt := runtime.New()
    exec, err := rt.Compile(b)
    if err != nil {
        panic(err)
    }
    defer exec.Close()

    // Run inference
    input := []float32{-1, 2, -3, 4, -5, 6}
    outputs, err := exec.Run(map[string]interface{}{"x": input})
    if err != nil {
        panic(err)
    }

    result := outputs["y"].([]float32)
    fmt.Println("Output:", result)
    // Output: [0 2 0 4 0 6]
}
```

### Available Operations

The MIL builder supports these operations:

- **Element-wise**: Add, Sub, Mul, Div, Neg, Abs
- **Activations**: Relu, Sigmoid, Tanh, Softmax
- **Math**: Exp, Log, Sqrt
- **Linear Algebra**: MatMul
- **Shape**: Reshape, Transpose
- **Reductions**: ReduceSum, ReduceMean, ReduceMax
- **Constants**: Const (for embedding literal values)

### Compute Unit Selection

Control which compute units are used:

```go
import "github.com/gomlx/go-coreml/internal/bridge"

// Use all available compute units (ANE + GPU + CPU)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeAll))

// CPU only (for debugging)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUOnly))

// CPU + GPU (skip Neural Engine)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUAndGPU))

// CPU + Neural Engine (skip GPU)
rt := runtime.New(runtime.WithComputeUnits(bridge.ComputeCPUAndANE))
```

## Project Structure

```
go-coreml/
├── internal/
│   └── bridge/          # Low-level cgo bindings to CoreML
│       ├── bridge.h     # C-compatible function declarations
│       ├── bridge.m     # Objective-C implementation
│       └── bridge.go    # cgo wrapper
├── model/
│   ├── builder.go       # MIL program builder
│   ├── ops.go           # MIL operation implementations
│   └── serialize.go     # Model serialization
├── runtime/
│   └── runtime.go       # High-level compilation and execution
├── proto/
│   └── coreml/
│       ├── milspec/     # Generated Go types from MIL.proto
│       ├── spec/        # Generated Go types from Model.proto
│       └── *.proto      # CoreML protobuf definitions
└── specs/
    └── 001-initial-plan.md  # Implementation plan
```

## Development

```bash
# Build
go build ./...

# Test
go test ./...

# Update protobuf files from coremltools
cd proto/coreml && ./update_protos.sh

# Regenerate Go code from protobufs
go generate ./...
```

## License

Apache 2.0 - see LICENSE file.

CoreML protobuf definitions are from [Apple's coremltools](https://github.com/apple/coremltools)
and are licensed under BSD-3-Clause.
