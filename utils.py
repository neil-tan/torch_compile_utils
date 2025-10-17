"""
Utility functions for benchmarking and analyzing compilation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import statistics
from dataclasses import dataclass
from .core import CompilationProfile, STANDARD_PROFILES


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    mode: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    compilation_time: float
    speedup: float = 1.0
    profile_used: Optional[str] = None
    cuda_graphs_enabled: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def benchmark_compilation(
    model: nn.Module,
    input_fn: Callable[[], Tuple[torch.Tensor, ...]],
    modes: List[str] = ["eager", "default", "reduce-overhead", "max-autotune"],
    num_warmup: int = 10,
    num_iterations: int = 100,
    verbose: bool = True
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark a model with different compilation modes.

    Args:
        model: Model to benchmark
        input_fn: Function that returns model inputs
        modes: List of compilation modes to test
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        verbose: Whether to print results

    Returns:
        Dictionary mapping mode names to benchmark results

    Example:
        def get_input():
            return torch.randn(32, 3, 224, 224).cuda()

        results = benchmark_compilation(
            model,
            get_input,
            modes=['eager', 'default', 'max-autotune']
        )
    """
    results = {}
    device = next(model.parameters()).device

    for mode in modes:
        if verbose:
            print(f"\nBenchmarking mode: {mode}")
            print("-" * 40)

        # Create a fresh copy of the model for each mode
        model_copy = model.__class__()
        model_copy.load_state_dict(model.state_dict())
        model_copy = model_copy.to(device)
        model_copy.eval()

        try:
            # Apply compilation
            compilation_start = time.perf_counter()

            if mode != "eager":
                if mode in STANDARD_PROFILES:
                    profile = STANDARD_PROFILES[mode]
                    compile_kwargs = profile.to_compile_kwargs()
                else:
                    compile_kwargs = {"mode": mode}

                model_copy = torch.compile(model_copy, **compile_kwargs)

            compilation_time = time.perf_counter() - compilation_start

            # Warmup
            if verbose:
                print(f"  Warming up ({num_warmup} iterations)...")

            with torch.no_grad():
                for _ in range(num_warmup):
                    inputs = input_fn()
                    if not isinstance(inputs, tuple):
                        inputs = (inputs,)
                    _ = model_copy(*inputs)

            # Synchronize CUDA
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # Benchmark
            if verbose:
                print(f"  Benchmarking ({num_iterations} iterations)...")

            times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    inputs = input_fn()
                    if not isinstance(inputs, tuple):
                        inputs = (inputs,)

                    if device.type == 'cuda':
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    _ = model_copy(*inputs)

                    if device.type == 'cuda':
                        torch.cuda.synchronize()

                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

            # Calculate statistics
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0
            min_time = min(times)
            max_time = max(times)

            results[mode] = BenchmarkResult(
                mode=mode,
                mean_time=mean_time,
                std_time=std_time,
                min_time=min_time,
                max_time=max_time,
                compilation_time=compilation_time,
                profile_used=mode if mode in STANDARD_PROFILES else None,
                cuda_graphs_enabled=mode == "max-autotune"
            )

            if verbose:
                print(f"  Mean: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
                print(f"  Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
                print(f"  Compilation time: {compilation_time:.2f}s")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")

            results[mode] = BenchmarkResult(
                mode=mode,
                mean_time=float('inf'),
                std_time=0,
                min_time=float('inf'),
                max_time=float('inf'),
                compilation_time=0,
                errors=[str(e)]
            )

    # Calculate speedups relative to eager
    if "eager" in results and results["eager"].mean_time > 0:
        eager_time = results["eager"].mean_time
        for mode, result in results.items():
            if result.mean_time < float('inf'):
                result.speedup = eager_time / result.mean_time

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Mode':<20} {'Mean (ms)':<12} {'Speedup':<10} {'Compile (s)':<12}")
        print("-" * 60)

        for mode in modes:
            if mode in results:
                r = results[mode]
                if r.mean_time < float('inf'):
                    print(
                        f"{mode:<20} {r.mean_time*1000:<12.2f} "
                        f"{r.speedup:<10.2f}x {r.compilation_time:<12.2f}"
                    )
                else:
                    print(f"{mode:<20} {'FAILED':<12} {'-':<10} {'-':<12}")

    return results


def compare_compilation_modes(
    model: nn.Module,
    test_inputs: List[torch.Tensor],
    modes: List[str] = ["default", "reduce-overhead", "max-autotune"],
    metrics: List[str] = ["throughput", "latency", "memory"]
) -> Dict[str, Dict[str, float]]:
    """
    Compare different compilation modes across multiple metrics.

    Args:
        model: Model to compare
        test_inputs: List of test inputs
        modes: Compilation modes to compare
        metrics: Metrics to measure

    Returns:
        Nested dict: {mode: {metric: value}}

    Example:
        comparison = compare_compilation_modes(
            model,
            [torch.randn(32, 3, 224, 224)],
            modes=['default', 'max-autotune']
        )
    """
    results = {}
    device = next(model.parameters()).device

    for mode in modes:
        results[mode] = {}

        # Create model copy
        model_copy = model.__class__()
        model_copy.load_state_dict(model.state_dict())
        model_copy = model_copy.to(device)

        # Compile
        if mode != "eager":
            compile_kwargs = {"mode": mode} if mode not in STANDARD_PROFILES else \
                STANDARD_PROFILES[mode].to_compile_kwargs()
            model_copy = torch.compile(model_copy, **compile_kwargs)

        # Measure metrics
        if "throughput" in metrics:
            # Measure samples/second
            batch_size = test_inputs[0].shape[0] if test_inputs else 1
            with torch.no_grad():
                start = time.perf_counter()
                for _ in range(100):
                    _ = model_copy(*test_inputs)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

            results[mode]["throughput"] = (100 * batch_size) / elapsed

        if "latency" in metrics:
            # Measure single-sample latency
            with torch.no_grad():
                times = []
                for _ in range(100):
                    start = time.perf_counter()
                    _ = model_copy(*test_inputs)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)

            results[mode]["latency"] = statistics.mean(times) * 1000  # ms

        if "memory" in metrics and device.type == 'cuda':
            # Measure peak memory usage
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model_copy(*test_inputs)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            results[mode]["memory"] = peak_memory

    return results


def get_optimal_settings(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    priorities: Dict[str, float] = {"speed": 0.7, "memory": 0.3},
    test_modes: List[str] = None
) -> Dict[str, Any]:
    """
    Automatically determine optimal compilation settings for a model.

    Args:
        model: Model to optimize
        sample_inputs: Sample inputs for testing
        priorities: Weights for different optimization goals
        test_modes: Modes to test (None = all standard modes)

    Returns:
        Dictionary with recommended settings

    Example:
        settings = get_optimal_settings(
            model,
            torch.randn(32, 3, 224, 224),
            priorities={"speed": 0.8, "memory": 0.2}
        )
        model = torch.compile(model, **settings['compile_kwargs'])
    """
    if test_modes is None:
        test_modes = ["default", "reduce-overhead", "max-autotune"]

    # Detect architecture
    from .architectures import detect_architecture
    arch_type = detect_architecture(model)

    # Get model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming fp32

    # Check for dynamic shapes
    has_dynamic_shapes = False
    if hasattr(model, "forward"):
        # Simple heuristic: run twice with different batch sizes
        try:
            with torch.no_grad():
                test1 = sample_inputs
                test2 = sample_inputs[:sample_inputs.shape[0]//2] if sample_inputs.shape[0] > 1 else sample_inputs
                _ = model(test1)
                _ = model(test2)
            has_dynamic_shapes = True
        except:
            has_dynamic_shapes = False

    # Run benchmarks
    def input_fn():
        return sample_inputs

    results = benchmark_compilation(
        model,
        input_fn,
        modes=test_modes,
        num_warmup=5,
        num_iterations=20,
        verbose=False
    )

    # Score each mode
    scores = {}
    for mode, result in results.items():
        if result.mean_time < float('inf'):
            # Speed score (lower is better, so invert)
            speed_score = 1.0 / result.mean_time if result.mean_time > 0 else 0

            # Memory score (placeholder - would need actual measurement)
            memory_score = 1.0  # Simplified

            # Combined score
            scores[mode] = (
                priorities.get("speed", 0.5) * speed_score +
                priorities.get("memory", 0.5) * memory_score
            )
        else:
            scores[mode] = 0

    # Find best mode
    best_mode = max(scores.keys(), key=lambda k: scores[k])
    best_result = results[best_mode]

    # Build recommendations
    recommendations = {
        "architecture": arch_type,
        "model_size_mb": model_size_mb,
        "has_dynamic_shapes": has_dynamic_shapes,
        "recommended_mode": best_mode,
        "expected_speedup": best_result.speedup,
        "compile_kwargs": {},
        "notes": []
    }

    # Set compile kwargs based on analysis
    if best_mode in STANDARD_PROFILES:
        recommendations["compile_kwargs"] = STANDARD_PROFILES[best_mode].to_compile_kwargs()
    else:
        recommendations["compile_kwargs"]["mode"] = best_mode

    # Architecture-specific adjustments
    if arch_type == "transformer":
        recommendations["compile_kwargs"]["fullgraph"] = not has_dynamic_shapes
        if not has_dynamic_shapes:
            recommendations["compile_kwargs"]["options"] = {"triton.cudagraphs": True}
            recommendations["notes"].append("CUDA graphs enabled for fixed shapes")

    elif arch_type == "rnn":
        recommendations["compile_kwargs"]["dynamic"] = True
        recommendations["notes"].append("Dynamic shapes enabled for RNN")

    elif arch_type == "cnn":
        recommendations["compile_kwargs"]["fullgraph"] = True
        recommendations["notes"].append("Full graph mode for CNN")

    # Size-based adjustments
    if model_size_mb > 1000:  # Large model
        recommendations["notes"].append("Consider gradient checkpointing for large model")

    if model_size_mb < 10:  # Small model
        recommendations["compile_kwargs"]["fullgraph"] = True
        recommendations["notes"].append("Full graph mode for small model")

    return recommendations


def analyze_compilation_failures(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze why compilation might be failing for a model.

    Args:
        model: Model to analyze
        sample_inputs: Sample inputs
        verbose: Whether to print analysis

    Returns:
        Dictionary with failure analysis

    Example:
        analysis = analyze_compilation_failures(model, inputs)
        if analysis['graph_breaks']:
            print(f"Found {len(analysis['graph_breaks'])} graph breaks")
    """
    analysis = {
        "graph_breaks": [],
        "dynamic_shapes": [],
        "unsupported_ops": [],
        "recommendations": []
    }

    # Try compilation with graph break detection
    try:
        import torch._dynamo

        # Enable graph break logging
        torch._dynamo.config.verbose = True
        torch._dynamo.config.log_level = "INFO"

        # Attempt compilation
        compiled = torch.compile(model, fullgraph=False)

        with torch.no_grad():
            _ = compiled(sample_inputs)

        # Check for graph breaks (would need to parse logs in real implementation)
        analysis["graph_breaks"].append("Check logs for torch._dynamo graph breaks")

    except Exception as e:
        analysis["compilation_error"] = str(e)

        # Parse error for common issues
        error_str = str(e).lower()
        if "dynamic" in error_str or "shape" in error_str:
            analysis["dynamic_shapes"].append("Dynamic shapes detected")
            analysis["recommendations"].append("Try mode='reduce-overhead' or dynamic=True")

        if "unsupported" in error_str:
            analysis["unsupported_ops"].append("Unsupported operations detected")
            analysis["recommendations"].append("Consider using fullgraph=False")

    # Check for common problematic patterns
    for name, module in model.named_modules():
        # Check for dynamic control flow
        if isinstance(module, nn.Module):
            # Simplified checks - would need more sophisticated analysis
            if "dropout" in name.lower() and module.training:
                analysis["recommendations"].append(f"Set {name} to eval mode for compilation")

    # Test shape consistency
    try:
        with torch.no_grad():
            # Test with different batch sizes
            if sample_inputs.ndim > 0:
                half_batch = sample_inputs[:sample_inputs.shape[0]//2]
                _ = model(half_batch)
                analysis["dynamic_shapes"].append("Model handles dynamic batch sizes")
    except:
        analysis["recommendations"].append("Model may require fixed input shapes")

    if verbose:
        print("Compilation Analysis:")
        print("=" * 60)

        if analysis.get("compilation_error"):
            print(f"Error: {analysis['compilation_error'][:200]}...")

        if analysis["graph_breaks"]:
            print(f"Graph breaks detected: {len(analysis['graph_breaks'])}")

        if analysis["dynamic_shapes"]:
            print(f"Dynamic shapes: {', '.join(analysis['dynamic_shapes'])}")

        if analysis["unsupported_ops"]:
            print(f"Unsupported ops: {', '.join(analysis['unsupported_ops'])}")

        if analysis["recommendations"]:
            print("\nRecommendations:")
            for rec in analysis["recommendations"]:
                print(f"  • {rec}")

    return analysis


def estimate_compilation_benefit(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    workload_type: str = "inference"
) -> Dict[str, Any]:
    """
    Estimate potential benefits from compilation.

    Args:
        model: Model to analyze
        sample_inputs: Sample inputs
        workload_type: Type of workload ('inference' or 'training')

    Returns:
        Dictionary with benefit estimates

    Example:
        benefits = estimate_compilation_benefit(model, inputs)
        if benefits['estimated_speedup'] > 1.5:
            print("Compilation highly recommended!")
    """
    # Get model characteristics
    total_params = sum(p.numel() for p in model.parameters())
    total_ops = 0  # Would need to actually count ops

    # Detect architecture
    from .architectures import detect_architecture
    arch_type = detect_architecture(model)

    # Base estimates by architecture
    speedup_estimates = {
        "transformer": {"inference": 2.5, "training": 1.8},
        "rnn": {"inference": 1.3, "training": 1.2},
        "cnn": {"inference": 2.0, "training": 1.6},
        "mlp": {"inference": 3.0, "training": 2.0},
        "unknown": {"inference": 1.5, "training": 1.3}
    }

    base_speedup = speedup_estimates.get(arch_type, speedup_estimates["unknown"])[workload_type]

    # Adjust based on model size
    if total_params > 1_000_000_000:  # >1B params
        base_speedup *= 0.8  # Large models benefit less
    elif total_params < 1_000_000:  # <1M params
        base_speedup *= 1.2  # Small models benefit more

    # Check for compilation-friendly patterns
    has_residual = False
    has_attention = False
    has_conv = False

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            has_attention = True
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            has_conv = True

    # Adjust estimates
    if has_attention:
        base_speedup *= 1.1  # Attention benefits from fusion

    if has_conv:
        base_speedup *= 1.05  # Convs benefit from optimization

    benefits = {
        "architecture": arch_type,
        "model_params": total_params,
        "workload_type": workload_type,
        "estimated_speedup": base_speedup,
        "confidence": "medium",
        "compilation_recommended": base_speedup > 1.3,
        "best_mode": "max-autotune" if base_speedup > 2.0 else "default",
        "notes": []
    }

    # Add notes
    if base_speedup > 2.0:
        benefits["notes"].append("High compilation benefit expected")
        benefits["confidence"] = "high"
    elif base_speedup > 1.5:
        benefits["notes"].append("Moderate compilation benefit expected")
    else:
        benefits["notes"].append("Limited compilation benefit expected")
        benefits["confidence"] = "low"

    if arch_type == "transformer":
        benefits["notes"].append("Consider Flash Attention for additional speedup")

    if workload_type == "training":
        benefits["notes"].append("Training compilation requires careful tuning")

    return benefits