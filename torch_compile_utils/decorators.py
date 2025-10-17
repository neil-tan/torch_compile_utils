"""
Method decorators for fine-grained compilation control.
"""

import torch
from functools import wraps
from typing import Optional, Callable, Any, List, Dict, Tuple
import time
import warnings
import os


def compile_method(profile: str = "default", cache_key: Optional[str] = None):
    """
    Decorator to compile a specific method with a given profile.

    Args:
        profile: Name of the compilation profile to use
        cache_key: Optional custom cache key (defaults to method name)

    Example:
        @compile_method(profile="static")
        def forward(self, x):
            return self.layers(x)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if compilation is disabled via environment variable
            if os.environ.get("TORCH_COMPILE_DISABLE", "").lower() in ["1", "true", "yes"]:
                return func(self, *args, **kwargs)

            # Get compilation config
            if hasattr(self, "get_profile"):
                compile_profile = self.get_profile(profile)
            else:
                # Fallback to default compilation
                compile_profile = None

            # Skip compilation in debug mode
            if compile_profile and compile_profile.mode is None:
                return func(self, *args, **kwargs)

            # Check if already compiled
            actual_cache_key = cache_key or f"_compiled_{profile}_{func.__name__}"

            if not hasattr(self, actual_cache_key):
                if compile_profile:
                    compile_kwargs = compile_profile.to_compile_kwargs()
                else:
                    compile_kwargs = {"mode": "default"}

                try:
                    compiled = torch.compile(func, **compile_kwargs)
                    setattr(self, actual_cache_key, compiled)
                except Exception as e:
                    warnings.warn(f"Compilation failed for {func.__name__}: {e}")
                    setattr(self, actual_cache_key, func)

            compiled_func = getattr(self, actual_cache_key)
            return compiled_func(self, *args, **kwargs)

        # Store metadata
        wrapper._is_compiled_method = True
        wrapper._compilation_profile = profile
        return wrapper
    return decorator


def compile_if(condition: Callable[..., bool], profile: str = "default"):
    """
    Conditionally compile a method based on runtime conditions.

    Args:
        condition: Callable that returns True when compilation should happen
        profile: Compilation profile to use when condition is met

    Example:
        @compile_if(lambda self, x: x.shape[0] > 32, profile="static")
        def process(self, x):
            return self.heavy_computation(x)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Evaluate condition
            should_compile = condition(self, *args, **kwargs)

            if should_compile:
                cache_key = f"_compiled_conditional_{profile}_{func.__name__}"

                if not hasattr(self, cache_key):
                    if hasattr(self, "get_profile"):
                        compile_profile = self.get_profile(profile)
                        if compile_profile and compile_profile.mode:
                            compile_kwargs = compile_profile.to_compile_kwargs()
                            compiled = torch.compile(func, **compile_kwargs)
                            setattr(self, cache_key, compiled)
                        else:
                            setattr(self, cache_key, func)
                    else:
                        compiled = torch.compile(func)
                        setattr(self, cache_key, compiled)

                compiled_func = getattr(self, cache_key)
                return compiled_func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        wrapper._is_conditional_compile = True
        wrapper._condition = condition
        return wrapper
    return decorator


def adaptive_compile(
    threshold_attr: str = "batch_size",
    threshold_value: int = 32,
    small_profile: str = "dynamic",
    large_profile: str = "static"
):
    """
    Adaptively compile based on runtime attributes.

    Args:
        threshold_attr: Attribute name to check
        threshold_value: Threshold value for switching profiles
        small_profile: Profile for values below threshold
        large_profile: Profile for values above threshold

    Example:
        @adaptive_compile(threshold_attr="sequence_length", threshold_value=512)
        def attention(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get attribute value
            current_value = getattr(self, threshold_attr, 0)

            # For tensor attributes, get the relevant dimension
            if torch.is_tensor(current_value):
                current_value = current_value.shape[0]
            elif hasattr(current_value, "__len__"):
                current_value = len(current_value)

            # Choose profile based on threshold
            profile = large_profile if current_value > threshold_value else small_profile
            cache_key = f"_compiled_adaptive_{profile}_{func.__name__}"

            if not hasattr(self, cache_key):
                if hasattr(self, "get_profile"):
                    compile_profile = self.get_profile(profile)
                    if compile_profile and compile_profile.mode:
                        compile_kwargs = compile_profile.to_compile_kwargs()
                        compiled = torch.compile(func, **compile_kwargs)
                        setattr(self, cache_key, compiled)
                    else:
                        setattr(self, cache_key, func)
                else:
                    compiled = torch.compile(func, mode="default")
                    setattr(self, cache_key, compiled)

            compiled_func = getattr(self, cache_key)
            return compiled_func(self, *args, **kwargs)

        wrapper._is_adaptive_compile = True
        wrapper._threshold_attr = threshold_attr
        return wrapper
    return decorator


def profile_guided(warmup_steps: int = 100, verbose: bool = False):
    """
    Profile-guided compilation that analyzes runtime patterns before compiling.

    Args:
        warmup_steps: Number of calls to profile before compiling
        verbose: Whether to print profiling information

    Example:
        @profile_guided(warmup_steps=50)
        def forward(self, x):
            return self.model(x)
    """
    def decorator(func):
        # Profiling state stored on the function itself
        profile_state = {
            "call_count": 0,
            "shape_variations": {},
            "time_samples": [],
            "compiled_func": None,
            "chosen_profile": None,
        }

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            state = profile_state

            if state["call_count"] < warmup_steps:
                # Profiling phase
                state["call_count"] += 1

                # Track shapes
                for i, arg in enumerate(args):
                    if torch.is_tensor(arg):
                        if i not in state["shape_variations"]:
                            state["shape_variations"][i] = set()
                        state["shape_variations"][i].add(arg.shape)

                # Time execution
                start_time = time.perf_counter()
                result = func(self, *args, **kwargs)
                elapsed = time.perf_counter() - start_time
                state["time_samples"].append(elapsed)

                # Compile after warmup
                if state["call_count"] == warmup_steps:
                    # Analyze profiling data
                    has_dynamic_shapes = any(
                        len(shapes) > 1
                        for shapes in state["shape_variations"].values()
                    )

                    # Choose profile based on analysis
                    if has_dynamic_shapes:
                        profile_name = "dynamic"
                    else:
                        profile_name = "static"

                    state["chosen_profile"] = profile_name

                    # Compile with chosen profile
                    if hasattr(self, "get_profile"):
                        compile_profile = self.get_profile(profile_name)
                        if compile_profile and compile_profile.mode:
                            compile_kwargs = compile_profile.to_compile_kwargs()
                        else:
                            compile_kwargs = {"mode": "default"}
                    else:
                        compile_kwargs = {"mode": "default", "dynamic": has_dynamic_shapes}

                    try:
                        state["compiled_func"] = torch.compile(func, **compile_kwargs)

                        if verbose:
                            avg_time = sum(state["time_samples"]) / len(state["time_samples"])
                            print(f"Profile-guided compilation for {func.__name__}:")
                            print(f"  Profile chosen: {profile_name}")
                            print(f"  Dynamic shapes: {has_dynamic_shapes}")
                            print(f"  Avg warmup time: {avg_time:.6f}s")
                            print(f"  Shape variations: {state['shape_variations']}")

                    except Exception as e:
                        warnings.warn(f"Profile-guided compilation failed: {e}")
                        state["compiled_func"] = func

                return result

            else:
                # Use compiled version
                if state["compiled_func"] is None:
                    state["compiled_func"] = func
                return state["compiled_func"](self, *args, **kwargs)

        wrapper._is_profile_guided = True
        wrapper._profile_state = profile_state
        return wrapper
    return decorator


def shape_specialized(*arg_indices: int):
    """
    Create specialized compilations for different tensor shapes.

    Args:
        arg_indices: Indices of arguments to specialize on

    Example:
        @shape_specialized(0, 1)  # Specialize on first two arguments
        def matmul(self, a, b):
            return torch.matmul(a, b)
    """
    def decorator(func):
        # Cache for different shape specializations
        specialization_cache = {}

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create shape signature
            shape_key = []
            for idx in arg_indices:
                if idx < len(args) and torch.is_tensor(args[idx]):
                    shape_key.append(args[idx].shape)
                else:
                    shape_key.append(None)
            shape_key = tuple(shape_key)

            # Get or create specialized compilation
            if shape_key not in specialization_cache:
                # Determine if all shapes are static
                all_static = all(s is not None for s in shape_key)

                if all_static:
                    # Static shapes - aggressive optimization
                    compile_kwargs = {
                        "mode": "max-autotune",
                        "fullgraph": True,
                        "options": {"triton.cudagraphs": True}
                    }
                else:
                    # Dynamic or mixed shapes
                    compile_kwargs = {"mode": "default", "dynamic": True}

                try:
                    specialization_cache[shape_key] = torch.compile(func, **compile_kwargs)
                except:
                    specialization_cache[shape_key] = func

            specialized_func = specialization_cache[shape_key]
            return specialized_func(self, *args, **kwargs)

        wrapper._is_shape_specialized = True
        wrapper._specialization_cache = specialization_cache
        return wrapper
    return decorator


def no_compile(func):
    """
    Decorator to explicitly prevent compilation of a method.

    Example:
        @no_compile
        def debug_forward(self, x):
            print(f"Debug: {x.shape}")
            return self.forward(x)
    """
    func._no_compile = True
    return func


def graph_break(func):
    """
    Force a graph break at this method.
    Useful for methods with Python control flow or debugging.

    Example:
        @graph_break
        def checkpoint(self):
            # Forces a graph break here
            return self.state.clone()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.compiler.is_compiling():
            torch._dynamo.graph_break()
        return func(*args, **kwargs)

    wrapper._has_graph_break = True
    return wrapper


def benchmark_method(num_warmup: int = 10, num_iterations: int = 100):
    """
    Decorator to benchmark a method's performance.

    Args:
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Example:
        @benchmark_method(num_warmup=5, num_iterations=50)
        def forward(self, x):
            return self.model(x)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Warmup
            for _ in range(num_warmup):
                _ = func(self, *args, **kwargs)

            # Synchronize if using CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                result = func(self, *args, **kwargs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                times.append(elapsed)

            # Store benchmark results
            if not hasattr(self, "_benchmark_results"):
                self._benchmark_results = {}

            self._benchmark_results[func.__name__] = {
                "times": times,
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }

            return result

        wrapper._is_benchmarked = True
        return wrapper
    return decorator