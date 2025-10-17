"""
Core compilation framework with mixins and base classes.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import warnings


class CompilationMode(Enum):
    """Standard compilation modes."""
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"
    MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"
    DEBUG = None  # No compilation


@dataclass
class CompilationProfile:
    """
    Compilation profile with all settings for torch.compile.

    Attributes:
        name: Profile name for identification
        mode: Compilation mode (default, reduce-overhead, max-autotune)
        fullgraph: Whether to compile as single graph
        dynamic: Whether to support dynamic shapes
        backend: Compilation backend (inductor, etc.)
        options: Additional compiler options
        condition: Optional condition for when to apply this profile
    """
    name: str = "default"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False
    backend: str = "inductor"
    options: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable[..., bool]] = None

    def to_compile_kwargs(self) -> Dict[str, Any]:
        """Convert to torch.compile keyword arguments."""
        kwargs = {}
        if self.mode:
            kwargs["mode"] = self.mode
        if self.fullgraph:
            kwargs["fullgraph"] = True
        if self.dynamic:
            kwargs["dynamic"] = True
        if self.backend != "inductor":
            kwargs["backend"] = self.backend
        if self.options:
            # If we have options, we can't also have mode in PyTorch 2.0+
            if "mode" in kwargs:
                del kwargs["mode"]
            kwargs["options"] = self.options
        return kwargs

    def merge_with(self, other: "CompilationProfile") -> "CompilationProfile":
        """Merge with another profile, other takes precedence."""
        return CompilationProfile(
            name=other.name if other.name != "default" else self.name,
            mode=other.mode if other.mode else self.mode,
            fullgraph=other.fullgraph or self.fullgraph,
            dynamic=other.dynamic or self.dynamic,
            backend=other.backend if other.backend != "inductor" else self.backend,
            options={**self.options, **other.options},
            condition=other.condition or self.condition,
        )


@dataclass
class CompileConfig:
    """
    Global compilation configuration for a model.

    This is the main configuration object that users interact with.
    """
    # Basic settings
    enabled: bool = True
    default_mode: str = "default"

    # CUDA graphs
    cuda_graphs_enabled: bool = True
    cuda_graph_threshold: int = 32

    # Profile-guided optimization
    profile_guided: bool = False
    profile_warmup_steps: int = 100

    # Dynamic shapes
    dynamic_shape_threshold: float = 0.1  # >10% variation = dynamic

    # Profiles
    profiles: Dict[str, CompilationProfile] = field(default_factory=dict)

    # Debug settings
    verbose: bool = False
    graph_breaks_allowed: bool = True

    def __post_init__(self):
        """Initialize default profiles if not provided."""
        if not self.profiles:
            self.profiles = self._get_default_profiles()

    def _get_default_profiles(self) -> Dict[str, CompilationProfile]:
        """Get default compilation profiles."""
        return {
            "static": CompilationProfile(
                name="static",
                mode="max-autotune",
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    "epilogue_fusion": True,
                }
            ),
            "dynamic": CompilationProfile(
                name="dynamic",
                mode="default",
                dynamic=True,
                options={"triton.cudagraphs": False}
            ),
            "inference": CompilationProfile(
                name="inference",
                mode="max-autotune",
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    "triton.autotune_pointwise": True,
                    "epilogue_fusion": True,
                }
            ),
            "training": CompilationProfile(
                name="training",
                mode="reduce-overhead",
                options={"triton.cudagraphs": False}
            ),
            "debug": CompilationProfile(
                name="debug",
                mode=None,  # No compilation
            )
        }


class CompilationConfigMixin:
    """
    Mixin that provides compilation configuration via class attributes.
    """

    # Default configuration (overridable in subclasses)
    COMPILE_CONFIG: CompileConfig = CompileConfig()

    # Class-level compilation cache
    _compilation_cache: Dict[str, Any] = {}
    _profile_stats: Dict[str, Dict] = {}

    @classmethod
    def get_compile_config(cls) -> CompileConfig:
        """Get the compilation configuration."""
        return cls.COMPILE_CONFIG

    @classmethod
    def set_compile_config(cls, config: CompileConfig):
        """Set the compilation configuration."""
        cls.COMPILE_CONFIG = config
        # Clear cache when config changes
        cls._compilation_cache.clear()

    @classmethod
    def get_profile(cls, name: str) -> Optional[CompilationProfile]:
        """Get a compilation profile by name."""
        return cls.COMPILE_CONFIG.profiles.get(name)

    @classmethod
    def add_profile(cls, profile: CompilationProfile):
        """Add or update a compilation profile."""
        cls.COMPILE_CONFIG.profiles[profile.name] = profile

    def prepare_for_compilation(self) -> None:
        """
        Prepare the model for compilation.
        This is called before compilation happens.
        """
        config = self.get_compile_config()

        if config.verbose:
            print(f"Preparing {self.__class__.__name__} for compilation")
            print(f"  Mode: {config.default_mode}")
            print(f"  CUDA graphs: {config.cuda_graphs_enabled}")
            print(f"  Profiles available: {list(config.profiles.keys())}")

        # Configure PyTorch settings
        if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
            if config.cuda_graphs_enabled:
                torch._inductor.config.triton.cudagraphs = True

            # Set other inductor configs
            if hasattr(torch._inductor.config, "aggressive_fusion"):
                torch._inductor.config.aggressive_fusion = True


class CompileAwareModel(nn.Module, CompilationConfigMixin):
    """
    Base class for models that support advanced compilation features.

    Inherits from both nn.Module and CompilationConfigMixin to provide
    full compilation awareness to any PyTorch model.
    """

    def __init__(self):
        super().__init__()
        self._compiled_methods = {}
        self._method_profiles = {}

    def compile_model(self, profile: str = "default") -> "CompileAwareModel":
        """
        Compile the entire model with the specified profile.

        Args:
            profile: Name of the compilation profile to use

        Returns:
            Self for chaining
        """
        compile_profile = self.get_profile(profile)
        if compile_profile is None:
            compile_profile = CompilationProfile(name=profile)

        if compile_profile.mode is None:
            # Debug mode - no compilation
            return self

        # Apply torch.compile
        compile_kwargs = compile_profile.to_compile_kwargs()
        compiled = torch.compile(self, **compile_kwargs)

        # Replace forward method
        self.forward = compiled.forward

        if self.COMPILE_CONFIG.verbose:
            print(f"Compiled {self.__class__.__name__} with profile '{profile}'")

        return self

    def get_compilation_info(self) -> Dict[str, Any]:
        """Get information about the model's compilation status."""
        return {
            "config": self.COMPILE_CONFIG,
            "compiled_methods": list(self._compiled_methods.keys()),
            "profiles": list(self.COMPILE_CONFIG.profiles.keys()),
            "cache_size": len(self._compilation_cache),
            "profile_stats": self._profile_stats,
        }

    def reset_compilation(self):
        """Reset all compilation state."""
        self._compiled_methods.clear()
        self._compilation_cache.clear()
        self._profile_stats.clear()

        # Reset forward if it was compiled
        if hasattr(self, "_forward_impl"):
            self.forward = self._forward_impl


# Predefined profiles for common use cases
STANDARD_PROFILES = {
    "transformer": CompilationProfile(
        name="transformer",
        mode="max-autotune",
        fullgraph=False,  # Transformers often have dynamic control flow
        options={
            "triton.cudagraphs": True,
            "epilogue_fusion": True,
            "coordinate_descent_tuning": True,
        }
    ),
    "rnn": CompilationProfile(
        name="rnn",
        mode="reduce-overhead",
        dynamic=True,
        options={
            "triton.cudagraphs": False,  # RNNs have dynamic shapes
        }
    ),
    "cnn": CompilationProfile(
        name="cnn",
        mode="max-autotune",
        fullgraph=True,  # CNNs usually have static graphs
        options={
            "triton.cudagraphs": True,
            "conv_1x1_as_mm": True,
            "epilogue_fusion": True,
        }
    ),
    "vision_transformer": CompilationProfile(
        name="vision_transformer",
        mode="max-autotune",
        fullgraph=True,  # ViTs have more static structure than LLMs
        options={
            "triton.cudagraphs": True,
            "epilogue_fusion": True,
            "coordinate_descent_tuning": True,
            "joint_graph_constant_folding": True,
        }
    ),
}