"""
Context managers for temporary compilation modes and settings.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
import time
import warnings
from .core import CompileConfig, CompilationProfile


@contextmanager
def optimization_scope(
    model: nn.Module,
    level: str = "default",
    verbose: bool = False
):
    """
    Temporary optimization scope with automatic restoration.

    Args:
        model: Model to optimize
        level: Optimization level ('debug', 'default', 'aggressive', 'max')
        verbose: Whether to print optimization info

    Example:
        with optimization_scope(model, level='max'):
            results = run_benchmark(model)
    """
    original_state = {}

    try:
        # Store original state
        if hasattr(model, "_original_forward"):
            original_state["forward"] = model.forward

        # Apply optimization based on level
        if level == "debug":
            # No compilation
            if verbose:
                print("Optimization scope: debug mode (no compilation)")
            yield model

        elif level == "default":
            # Standard compilation
            if verbose:
                print("Optimization scope: default compilation")
            model.forward = torch.compile(model.forward, mode="default")
            yield model

        elif level == "aggressive":
            # Aggressive optimization
            if verbose:
                print("Optimization scope: aggressive optimization")
            model.forward = torch.compile(
                model.forward,
                mode="max-autotune",
                fullgraph=True
            )
            yield model

        elif level == "max":
            # Maximum optimization with CUDA graphs
            if verbose:
                print("Optimization scope: maximum optimization with CUDA graphs")

            # Configure inductor for maximum performance
            if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
                orig_cudagraphs = torch._inductor.config.triton.cudagraphs
                torch._inductor.config.triton.cudagraphs = True
                original_state["cudagraphs"] = orig_cudagraphs

            model.forward = torch.compile(
                model.forward,
                mode="max-autotune",
                fullgraph=True,
                options={"triton.cudagraphs": True}
            )
            yield model

        else:
            raise ValueError(f"Unknown optimization level: {level}")

    finally:
        # Restore original state
        if "forward" in original_state:
            model.forward = original_state["forward"]

        if "cudagraphs" in original_state:
            if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
                torch._inductor.config.triton.cudagraphs = original_state["cudagraphs"]

        if verbose:
            print(f"Exited optimization scope: {level}")


@contextmanager
def cuda_graph_mode(enabled: bool = True, threshold: int = 32):
    """
    Temporarily enable or disable CUDA graphs.

    Args:
        enabled: Whether to enable CUDA graphs
        threshold: Minimum batch size for CUDA graphs

    Example:
        with cuda_graph_mode(enabled=False):
            # Run without CUDA graphs
            output = model(dynamic_input)
    """
    if not torch.cuda.is_available():
        yield
        return

    original_state = {}

    try:
        # Store and modify inductor config
        if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
            original_state["cudagraphs"] = torch._inductor.config.triton.cudagraphs
            original_state["cudagraph_trees"] = getattr(
                torch._inductor.config.triton, "cudagraph_trees", False
            )

            torch._inductor.config.triton.cudagraphs = enabled
            if hasattr(torch._inductor.config.triton, "cudagraph_trees"):
                torch._inductor.config.triton.cudagraph_trees = enabled

            # Set threshold
            if hasattr(torch._inductor.config, "triton.cudagraph_skip_dynamic_graphs"):
                original_state["skip_dynamic"] = torch._inductor.config.triton.cudagraph_skip_dynamic_graphs
                torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = not enabled

        yield

    finally:
        # Restore original state
        if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
            if "cudagraphs" in original_state:
                torch._inductor.config.triton.cudagraphs = original_state["cudagraphs"]

            if "cudagraph_trees" in original_state:
                if hasattr(torch._inductor.config.triton, "cudagraph_trees"):
                    torch._inductor.config.triton.cudagraph_trees = original_state["cudagraph_trees"]

            if "skip_dynamic" in original_state:
                if hasattr(torch._inductor.config, "triton.cudagraph_skip_dynamic_graphs"):
                    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = original_state["skip_dynamic"]


@contextmanager
def eager_mode():
    """
    Temporarily disable all compilation (run in eager mode).

    Useful for debugging or when compilation causes issues.

    Example:
        with eager_mode():
            # Debug model behavior
            output = model(input)
            print(f"Intermediate: {model.intermediate_value}")
    """
    original_state = {}

    try:
        # Disable torch.compile
        if hasattr(torch, "_dynamo"):
            original_state["dynamo_disable"] = torch._dynamo.config.disable
            torch._dynamo.config.disable = True

        # Disable torch.jit.script
        if hasattr(torch.jit, "_enabled"):
            original_state["jit_enabled"] = torch.jit._enabled
            torch.jit._enabled = False

        yield

    finally:
        # Restore original state
        if "dynamo_disable" in original_state:
            torch._dynamo.config.disable = original_state["dynamo_disable"]

        if "jit_enabled" in original_state:
            torch.jit._enabled = original_state["jit_enabled"]


@contextmanager
def profiling_context(
    warmup_steps: int = 10,
    profile_steps: int = 100,
    print_results: bool = True
):
    """
    Context manager that profiles execution and compilation.

    Args:
        warmup_steps: Number of warmup iterations
        profile_steps: Number of profiled iterations
        print_results: Whether to print profiling results

    Example:
        with profiling_context(warmup_steps=5, profile_steps=20) as prof:
            for batch in dataloader:
                output = model(batch)
                prof.step()
    """

    class Profiler:
        def __init__(self):
            self.warmup_steps = warmup_steps
            self.profile_steps = profile_steps
            self.current_step = 0
            self.times = []
            self.compilation_time = 0
            self.is_warmup = True
            self.start_time = None

        def step(self):
            """Call after each iteration."""
            self.current_step += 1

            if self.current_step == 1:
                self.start_time = time.perf_counter()

            if self.current_step == self.warmup_steps:
                self.is_warmup = False
                self.compilation_time = time.perf_counter() - self.start_time
                self.start_time = time.perf_counter()

            if not self.is_warmup and self.current_step <= self.warmup_steps + self.profile_steps:
                elapsed = time.perf_counter() - self.start_time
                self.times.append(elapsed)
                self.start_time = time.perf_counter()

        def get_results(self):
            """Get profiling results."""
            if not self.times:
                return {}

            return {
                "compilation_time": self.compilation_time,
                "avg_iteration_time": sum(self.times) / len(self.times),
                "min_iteration_time": min(self.times),
                "max_iteration_time": max(self.times),
                "total_time": self.compilation_time + sum(self.times),
                "iterations_profiled": len(self.times),
            }

    profiler = Profiler()

    try:
        yield profiler

    finally:
        if print_results:
            results = profiler.get_results()
            if results:
                print("\nProfiling Results:")
                print("-" * 40)
                print(f"Compilation time: {results['compilation_time']:.3f}s")
                print(f"Avg iteration: {results['avg_iteration_time']*1000:.2f}ms")
                print(f"Min iteration: {results['min_iteration_time']*1000:.2f}ms")
                print(f"Max iteration: {results['max_iteration_time']*1000:.2f}ms")
                print(f"Total time: {results['total_time']:.3f}s")


@contextmanager
def experiment_mode(
    model: nn.Module,
    compile_settings: Dict[str, Any],
    baseline_settings: Optional[Dict[str, Any]] = None,
    compare: bool = True
):
    """
    Experimental compilation mode with automatic comparison.

    Args:
        model: Model to experiment with
        compile_settings: Compilation settings to test
        baseline_settings: Baseline settings for comparison
        compare: Whether to run comparison

    Example:
        with experiment_mode(
            model,
            compile_settings={'mode': 'max-autotune'},
            baseline_settings={'mode': 'default'}
        ) as exp:
            results = benchmark(model)
            exp.record_metric('throughput', results.throughput)
    """

    class Experiment:
        def __init__(self):
            self.metrics = {}
            self.baseline_metrics = {}
            self.compile_settings = compile_settings
            self.baseline_settings = baseline_settings

        def record_metric(self, name: str, value: float):
            """Record a metric for the experiment."""
            self.metrics[name] = value

        def record_baseline_metric(self, name: str, value: float):
            """Record a baseline metric."""
            self.baseline_metrics[name] = value

        def get_comparison(self) -> Dict[str, Any]:
            """Get comparison between experiment and baseline."""
            if not self.baseline_metrics:
                return self.metrics

            comparison = {}
            for name, value in self.metrics.items():
                if name in self.baseline_metrics:
                    baseline = self.baseline_metrics[name]
                    comparison[name] = {
                        "experiment": value,
                        "baseline": baseline,
                        "improvement": (value - baseline) / baseline * 100
                    }
                else:
                    comparison[name] = {"experiment": value}

            return comparison

    experiment = Experiment()
    original_forward = model.forward

    try:
        # Apply experimental compilation
        model.forward = torch.compile(model.forward, **compile_settings)

        yield experiment

    finally:
        # Restore original
        model.forward = original_forward

        # Run baseline comparison if requested
        if compare and baseline_settings:
            model.forward = torch.compile(model.forward, **baseline_settings)
            # Note: Actual baseline execution would happen here in real usage

        # Print comparison
        if compare and experiment.metrics:
            comparison = experiment.get_comparison()
            print("\nExperiment Results:")
            print("-" * 40)
            for name, data in comparison.items():
                if isinstance(data, dict) and "improvement" in data:
                    print(f"{name}:")
                    print(f"  Experiment: {data['experiment']:.4f}")
                    print(f"  Baseline: {data['baseline']:.4f}")
                    print(f"  Improvement: {data['improvement']:+.1f}%")
                else:
                    print(f"{name}: {data}")


@contextmanager
def device_scope(device: str):
    """
    Temporarily set the default device for tensor creation.

    Args:
        device: Device to use ('cuda', 'cpu', 'cuda:0', etc.)

    Example:
        with device_scope('cuda:1'):
            # All tensors created here go to cuda:1
            x = torch.randn(100, 100)
    """
    original_device = torch.cuda.current_device() if torch.cuda.is_available() else None

    try:
        if device.startswith('cuda'):
            if torch.cuda.is_available():
                device_id = int(device.split(':')[1]) if ':' in device else 0
                torch.cuda.set_device(device_id)

        # Set default tensor type based on device
        if device == 'cpu':
            torch.set_default_tensor_type(torch.FloatTensor)
        elif device.startswith('cuda'):
            if torch.cuda.is_available():
                torch.set_default_tensor_type(torch.cuda.FloatTensor)

        yield

    finally:
        # Restore original device
        if original_device is not None and torch.cuda.is_available():
            torch.cuda.set_device(original_device)

        # Reset default tensor type
        torch.set_default_tensor_type(torch.FloatTensor)


@contextmanager
def compilation_override(model: nn.Module, profile: CompilationProfile):
    """
    Override compilation settings temporarily.

    Args:
        model: Model to override settings for
        profile: Compilation profile to apply

    Example:
        static_profile = CompilationProfile(
            name="static_test",
            mode="max-autotune",
            fullgraph=True
        )

        with compilation_override(model, static_profile):
            # Model uses static profile here
            output = model(fixed_size_input)
    """
    # Store original config if model has one
    original_config = None
    if hasattr(model, 'COMPILE_CONFIG'):
        original_config = model.COMPILE_CONFIG

    try:
        # Apply temporary profile
        if hasattr(model, 'COMPILE_CONFIG'):
            # Temporarily modify existing config
            temp_config = CompileConfig(
                enabled=True,
                profiles={"temp": profile}
            )
            model.COMPILE_CONFIG = temp_config

        # Apply compilation with new profile
        if profile.mode:
            compiled_forward = torch.compile(model.forward, **profile.to_compile_kwargs())
            original_forward = model.forward
            model.forward = compiled_forward

        yield model

    finally:
        # Restore original
        if original_config is not None:
            model.COMPILE_CONFIG = original_config

        if 'original_forward' in locals():
            model.forward = original_forward


@contextmanager
def shape_tracing(verbose: bool = True):
    """
    Context manager to trace tensor shapes during execution.

    Useful for identifying dynamic shapes that prevent CUDA graph optimization.

    Args:
        verbose: Whether to print shape information

    Example:
        with shape_tracing() as tracer:
            for batch in dataloader:
                output = model(batch)

            # Analyze shape variations
            variations = tracer.get_shape_variations()
    """

    class ShapeTracer:
        def __init__(self):
            self.shapes = {}
            self.shape_counts = {}
            self.original_new = torch.Tensor.__new__

        def __enter__(self):
            # Hook into tensor creation
            def traced_new(cls, *args, **kwargs):
                tensor = self.original_new(cls, *args, **kwargs)

                # Record shape
                shape = tuple(tensor.shape)
                if shape not in self.shapes:
                    self.shapes[shape] = 0
                self.shapes[shape] += 1

                return tensor

            torch.Tensor.__new__ = traced_new
            return self

        def __exit__(self, *args):
            # Restore original
            torch.Tensor.__new__ = self.original_new

        def get_shape_variations(self) -> Dict[str, List[Tuple[int, ...]]]:
            """Group shapes by dimensionality."""
            variations = {}

            for shape in self.shapes:
                dim = len(shape)
                if dim not in variations:
                    variations[dim] = []
                variations[dim].append(shape)

            return variations

        def print_summary(self):
            """Print shape analysis summary."""
            variations = self.get_shape_variations()

            print("\nShape Analysis:")
            print("-" * 40)

            for dim, shapes in sorted(variations.items()):
                print(f"\n{dim}D tensors: {len(shapes)} unique shapes")

                # Show top 5 most common
                sorted_shapes = sorted(
                    shapes,
                    key=lambda s: self.shapes[s],
                    reverse=True
                )[:5]

                for shape in sorted_shapes:
                    count = self.shapes[shape]
                    print(f"  {shape}: {count} occurrences")

            # Identify problematic dynamic shapes
            dynamic_dims = set()
            for shapes in variations.values():
                if len(shapes) > 1:
                    # Find which dimensions vary
                    for i in range(len(shapes[0])):
                        values = set(s[i] for s in shapes if i < len(s))
                        if len(values) > 1:
                            dynamic_dims.add(i)

            if dynamic_dims:
                print(f"\nDynamic dimensions detected: {dynamic_dims}")
                print("These may prevent CUDA graph optimization")

    tracer = ShapeTracer()

    try:
        with tracer:
            yield tracer
    finally:
        if verbose:
            tracer.print_summary()


# Convenience function for nested contexts
def multi_context(*managers):
    """
    Combine multiple context managers.

    Example:
        with multi_context(
            cuda_graph_mode(True),
            optimization_scope(model, 'max'),
            profiling_context()
        ) as (cuda_ctx, opt_ctx, prof):
            # All contexts active here
            pass
    """
    from contextlib import ExitStack

    with ExitStack() as stack:
        contexts = [stack.enter_context(cm) for cm in managers]
        yield contexts if len(contexts) > 1 else contexts[0]