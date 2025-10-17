"""
torch_compile_utils: High-level compilation orchestration for PyTorch 2.0+

A comprehensive framework for optimizing PyTorch models with torch.compile,
providing architecture-aware compilation, decorators, context managers, and
profile-guided optimization.
"""

__version__ = "0.1.0"

# Core imports
from .core import (
    CompilationProfile,
    CompileConfig,
    CompileAwareModel,
)

# Decorators
from .decorators import (
    compile_method,
    compile_if,
    adaptive_compile,
    profile_guided,
    shape_specialized,
    no_compile,
    graph_break,
)

# Architecture-specific
from .architectures import (
    ParallelContextModel,
    SequentialModel,
    auto_optimize,
    detect_architecture,
)

# Context managers
from .context_managers import (
    optimization_scope,
    cuda_graph_mode,
    eager_mode,
    profiling_context,
    experiment_mode,
    device_scope,
)

# Utilities
from .utils import (
    benchmark_compilation,
    compare_compilation_modes,
    get_optimal_settings,
)

__all__ = [
    # Core
    "CompilationProfile",
    "CompileConfig",
    "CompileAwareModel",

    # Decorators
    "compile_method",
    "compile_if",
    "adaptive_compile",
    "profile_guided",
    "shape_specialized",
    "no_compile",
    "graph_break",

    # Architectures
    "ParallelContextModel",
    "SequentialModel",
    "auto_optimize",
    "detect_architecture",

    # Context managers
    "optimization_scope",
    "cuda_graph_mode",
    "eager_mode",
    "profiling_context",
    "experiment_mode",
    "device_scope",

    # Utils
    "benchmark_compilation",
    "compare_compilation_modes",
    "get_optimal_settings",
]