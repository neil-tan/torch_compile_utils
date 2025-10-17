"""
Architecture-specific compilation optimizations.
"""

import torch
import torch.nn as nn
from typing import Optional, Type, Dict, Any, Tuple
from .core import CompileAwareModel, CompileConfig, CompilationProfile, STANDARD_PROFILES


class ParallelContextModel(CompileAwareModel):
    """
    Base class for models that process entire context windows in parallel.
    Optimized for Transformers and attention-based architectures.

    Features:
    - Fixed-size tensor operations for CUDA graph optimization
    - Padding-aware processing
    - Optimized for parallel computation
    """

    def __init__(self):
        super().__init__()

        # Override default config for parallel architectures
        self.COMPILE_CONFIG = CompileConfig(
            enabled=True,
            default_mode="max-autotune",
            cuda_graphs_enabled=True,
            cuda_graph_threshold=32,
            profiles={
                **self.COMPILE_CONFIG.profiles,
                "default": STANDARD_PROFILES["transformer"],
            }
        )

    def prepare_for_compilation(self) -> None:
        """Prepare parallel model for compilation."""
        super().prepare_for_compilation()

        # Enable specific optimizations for parallel processing
        import torch as torch_module  # Use different name to avoid shadowing
        if hasattr(torch_module, "_inductor") and hasattr(torch_module._inductor, "config"):
            import torch._inductor.config

            # Enable CUDA graphs for fixed shapes
            torch._inductor.config.triton.cudagraphs = True

            # Enable aggressive fusion for attention patterns
            if hasattr(torch._inductor.config, "aggressive_fusion"):
                torch._inductor.config.aggressive_fusion = True

            # Enable coordinate descent tuning for attention
            if hasattr(torch._inductor.config, "coordinate_descent_tuning"):
                torch._inductor.config.coordinate_descent_tuning = True

        # Enable Flash Attention if available
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

        if self.COMPILE_CONFIG.verbose:
            print("✓ Configured for parallel/transformer architecture")


class SequentialModel(CompileAwareModel):
    """
    Base class for models that process sequences step-by-step.
    Optimized for RNNs, LSTMs, GRUs, and State Space Models.

    Features:
    - Dynamic shape support
    - State management optimization
    - Sequential processing patterns
    """

    def __init__(self):
        super().__init__()

        # Override default config for sequential architectures
        self.COMPILE_CONFIG = CompileConfig(
            enabled=True,
            default_mode="reduce-overhead",  # Faster compilation for dynamic
            cuda_graphs_enabled=False,  # Disable due to dynamic shapes
            dynamic_shape_threshold=0.5,  # More tolerant of variations
            profiles={
                **self.COMPILE_CONFIG.profiles,
                "default": STANDARD_PROFILES["rnn"],
            }
        )

    def prepare_for_compilation(self) -> None:
        """Prepare sequential model for compilation."""
        super().prepare_for_compilation()

        # Disable CUDA graphs for dynamic shapes
        if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
            import torch._inductor.config
            torch._inductor.config.triton.cudagraphs = False

            # Enable dynamic shape support
            if hasattr(torch._dynamo, "config"):
                torch._dynamo.config.capture_dynamic_output_shape_ops = True

        if self.COMPILE_CONFIG.verbose:
            print("✓ Configured for sequential/RNN architecture")


class ConvolutionalModel(CompileAwareModel):
    """
    Base class for CNN-based models.
    Optimized for convolutional neural networks.

    Features:
    - Conv-specific optimizations
    - Channel-last memory format
    - Fusion opportunities
    """

    def __init__(self):
        super().__init__()

        self.COMPILE_CONFIG = CompileConfig(
            enabled=True,
            default_mode="max-autotune",
            cuda_graphs_enabled=True,
            profiles={
                **self.COMPILE_CONFIG.profiles,
                "default": STANDARD_PROFILES["cnn"],
            }
        )

    def prepare_for_compilation(self) -> None:
        """Prepare CNN model for compilation."""
        super().prepare_for_compilation()

        # Enable conv-specific optimizations
        if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
            import torch._inductor.config

            # Enable convolution optimizations
            if hasattr(torch._inductor.config, "conv_1x1_as_mm"):
                torch._inductor.config.conv_1x1_as_mm = True

            # Enable NHWC (channels-last) format for better performance
            if hasattr(torch._inductor.config, "force_channels_last"):
                torch._inductor.config.force_channels_last = True

        # Convert model to channels-last format
        try:
            self.to(memory_format=torch.channels_last)
            if self.COMPILE_CONFIG.verbose:
                print("✓ Converted to channels-last format")
        except:
            pass  # Not all models support channels-last

        if self.COMPILE_CONFIG.verbose:
            print("✓ Configured for CNN architecture")


def detect_architecture(model: nn.Module) -> str:
    """
    Automatically detect the architecture type of a model.

    Args:
        model: PyTorch model to analyze

    Returns:
        Architecture type: "transformer", "rnn", "cnn", "hybrid", or "unknown"
    """
    module_counts = {
        "attention": 0,
        "rnn": 0,
        "conv": 0,
        "linear": 0,
    }

    # Count different module types
    for module in model.modules():
        if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            module_counts["attention"] += 1
        elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            module_counts["rnn"] += 1
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module_counts["conv"] += 1
        elif isinstance(module, nn.Linear):
            module_counts["linear"] += 1

    # Determine architecture based on module composition
    total_modules = sum(module_counts.values())
    if total_modules == 0:
        return "unknown"

    # Calculate ratios
    attention_ratio = module_counts["attention"] / total_modules
    rnn_ratio = module_counts["rnn"] / total_modules
    conv_ratio = module_counts["conv"] / total_modules

    # Classify based on dominant module type
    if attention_ratio > 0.1:
        return "transformer"
    elif rnn_ratio > 0.1:
        return "rnn"
    elif conv_ratio > 0.2:
        return "cnn"
    elif attention_ratio > 0 and conv_ratio > 0:
        return "vision_transformer"
    elif max(module_counts.values()) == module_counts["linear"]:
        return "mlp"
    else:
        return "hybrid"


def auto_optimize(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    Automatically optimize a model based on its architecture.

    Args:
        model: PyTorch model to optimize
        verbose: Whether to print optimization details

    Returns:
        Optimized model with appropriate compilation settings
    """
    # Check environment variable first
    import os
    if os.environ.get("TORCH_COMPILE_DISABLE", "").lower() in ["1", "true", "yes"]:
        if verbose:
            print("⚠️ torch.compile disabled via TORCH_COMPILE_DISABLE environment variable")
        return model

    # Detect architecture
    arch_type = detect_architecture(model)

    if verbose:
        print(f"Detected architecture: {arch_type}")

    # Get appropriate profile
    if arch_type in STANDARD_PROFILES:
        profile = STANDARD_PROFILES[arch_type]
    else:
        # Default profile for unknown architectures
        profile = CompilationProfile(
            name="auto",
            mode="default",
            dynamic=True,  # Safe default
        )

    # Apply compilation
    if profile.mode:
        try:
            compiled_model = torch.compile(model, **profile.to_compile_kwargs())
            if verbose:
                print(f"✓ Applied {arch_type} optimization profile")
                print(f"  Mode: {profile.mode}")
                print(f"  CUDA graphs: {profile.options.get('triton.cudagraphs', False)}")
            return compiled_model
        except Exception as e:
            if verbose:
                print(f"⚠ Compilation failed: {e}")
                print("  Returning uncompiled model")
            return model
    else:
        if verbose:
            print("  No compilation applied (debug mode)")
        return model


def get_architecture_recommendations(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed recommendations for optimizing a model.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary with optimization recommendations
    """
    arch_type = detect_architecture(model)

    # Count parameters and layers
    total_params = sum(p.numel() for p in model.parameters())
    total_layers = len(list(model.modules()))

    recommendations = {
        "architecture": arch_type,
        "total_parameters": total_params,
        "total_layers": total_layers,
        "recommendations": [],
        "profile": None,
        "warnings": [],
    }

    # Architecture-specific recommendations
    if arch_type == "transformer":
        recommendations["profile"] = STANDARD_PROFILES["transformer"]
        recommendations["recommendations"].extend([
            "Use Flash Attention if sequence length > 512",
            "Enable CUDA graphs for fixed sequence lengths",
            "Consider gradient checkpointing if memory-constrained",
            "Use mixed precision (fp16/bf16) for better performance",
        ])

        # Check for specific optimizations
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                if not hasattr(module, "batch_first") or not module.batch_first:
                    recommendations["warnings"].append(
                        "Consider using batch_first=True for better performance"
                    )

    elif arch_type == "rnn":
        recommendations["profile"] = STANDARD_PROFILES["rnn"]
        recommendations["recommendations"].extend([
            "Disable CUDA graphs due to dynamic shapes",
            "Consider using packed sequences for variable lengths",
            "Use torch.jit.script for RNN cells if possible",
            "Enable dynamic shape compilation",
        ])

    elif arch_type == "cnn":
        recommendations["profile"] = STANDARD_PROFILES["cnn"]
        recommendations["recommendations"].extend([
            "Use channels-last memory format (NHWC)",
            "Enable conv-GEMM fusion for 1x1 convolutions",
            "Consider using torch.fx for graph optimization",
            "Enable CUDA graphs for fixed input sizes",
        ])

    elif arch_type == "vision_transformer":
        recommendations["profile"] = STANDARD_PROFILES["vision_transformer"]
        recommendations["recommendations"].extend([
            "Combine CNN and Transformer optimizations",
            "Use channels-last for conv layers",
            "Enable Flash Attention for transformer blocks",
            "Consider patch merging optimizations",
        ])

    # Size-based recommendations
    if total_params > 1_000_000_000:  # 1B+ parameters
        recommendations["recommendations"].append(
            "Large model: Consider model parallelism or gradient checkpointing"
        )
    elif total_params < 1_000_000:  # <1M parameters
        recommendations["recommendations"].append(
            "Small model: Enable fullgraph=True for better optimization"
        )

    return recommendations