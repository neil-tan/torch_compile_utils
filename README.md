# torch_compile_utils

A high-level compilation orchestration framework for PyTorch 2.0+ that provides architecture-aware optimization, decorators, context managers, and profile-guided compilation.

## Features

- **Architecture-aware compilation** - Automatically detects and optimizes for Transformers, RNNs, CNNs
- **Method-level decorators** - Fine-grained control with `@compile_method`, `@adaptive_compile`, `@profile_guided`
- **Context managers** - Temporary compilation modes with `optimization_scope`, `cuda_graph_mode`, `eager_mode`
- **Profile-guided optimization** - Learn from runtime patterns before compiling
- **Shape specialization** - Create optimized kernels for specific tensor shapes
- **Benchmarking utilities** - Compare compilation modes and measure performance

## Quick Start

```python
from torch_compile_utils import (
    CompileAwareModel,
    compile_method,
    optimization_scope,
    auto_optimize
)

# 1. Automatic optimization based on architecture
model = auto_optimize(your_model)

# 2. Class-based compilation with decorators
class MyModel(CompileAwareModel):
    @compile_method(profile="static")
    def forward(self, x):
        return self.layers(x)

# 3. Context managers for temporary modes
with optimization_scope(model, level="max"):
    results = benchmark(model)

# 4. Architecture-specific base classes
from torch_compile_utils import ParallelContextModel

class MyTransformer(ParallelContextModel):
    # Automatically configured for transformer architectures
    pass
```

## Installation

Currently this is an internal package. To use it:

```python
import sys
sys.path.insert(0, "path/to/external")
from torch_compile_utils import ...
```

## Core Components

### 1. Compilation Profiles

Pre-configured profiles for different scenarios:
- `static` - Fixed shapes, maximum optimization
- `dynamic` - Variable shapes, flexible compilation
- `inference` - Optimized for inference workloads
- `training` - Balanced for training loops
- `debug` - No compilation for debugging

### 2. Architecture-Specific Models

Base classes optimized for different architectures:
- `ParallelContextModel` - Transformers and attention-based models (CUDA graphs enabled)
- `SequentialModel` - RNNs, LSTMs, state space models (dynamic shapes)
- `ConvolutionalModel` - CNNs with channel-last optimization

### 3. Decorators

Method-level compilation control:
- `@compile_method` - Compile specific methods
- `@compile_if` - Conditional compilation based on runtime
- `@adaptive_compile` - Switch profiles based on input characteristics
- `@profile_guided` - Learn from warmup before compiling
- `@shape_specialized` - Create shape-specific kernels

### 4. Context Managers

Temporary compilation modes:
- `optimization_scope` - Temporary optimization levels
- `cuda_graph_mode` - Enable/disable CUDA graphs
- `eager_mode` - Disable all compilation
- `profiling_context` - Profile execution with warmup
- `experiment_mode` - Compare compilation strategies

## Examples

### Adaptive Compilation

```python
class Model(CompileAwareModel):
    @adaptive_compile(
        threshold_attr="batch_size",
        threshold_value=32,
        small_profile="dynamic",
        large_profile="static"
    )
    def forward(self, x):
        self.batch_size = x.shape[0]
        return self.process(x)
```

### Profile-Guided Optimization

```python
@profile_guided(warmup_steps=100)
def forward(self, x):
    # Analyzes shapes and patterns during warmup
    # Then compiles with optimal settings
    return self.model(x)
```

### Mixed Patterns

```python
class HybridModel(CompileAwareModel):
    @compile_method(profile="static")
    def conv_layers(self, x):
        # CNN processing with static compilation
        return self.convs(x)

    @adaptive_compile(threshold_attr="seq_len")
    def attention_layers(self, x):
        # Transformer processing with adaptive compilation
        return self.transformer(x)
```

## Performance

Typical speedups (varies by model and hardware):
- Transformers: 2.0-3.0x
- CNNs: 1.8-2.5x
- RNNs: 1.2-1.5x
- Small models (< 1M params): 2.5-3.5x
- Large models (> 1B params): 1.5-2.0x

## Testing

```bash
python -m pytest tests/test_torch_compile_utils.py
```

## Benchmarking

```bash
python benchmarks/benchmark_torch_compile_utils.py
```

## Requirements

- PyTorch >= 2.0.0
- Python >= 3.8
- CUDA (optional, for GPU optimizations)

## Future Work

- Integration with PyTorch core as `torch.compile.utils`
- Additional architecture patterns (GNNs, Diffusion models)
- Automatic hyperparameter tuning for compilation
- Distributed compilation strategies