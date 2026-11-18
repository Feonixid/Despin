import time
from typing import Any, Tuple, Optional
import torch


class KVCacheManager:
    """
    Manages in-memory KV-Cache states with O(1) rollback slicing.
    Supports both modern DynamicCache and legacy tuple-of-tensors structures.
    """

    @staticmethod
    def get_seq_length(past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
            return past_key_values.layers[0].keys.shape[2]
        if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            return past_key_values[0][0].shape[2]
        return 0

    @staticmethod
    def rollback(
        past_key_values: Any,
        target_seq_len: int,
    ) -> Tuple[Any, float]:
        """
        Physically rolls back the KV cache to `target_seq_len`.
        Returns (rolled_back_cache, rollback_latency_ms).
        """
        start_time = time.perf_counter()
        if past_key_values is None:
            return None, 0.0

        current_len = KVCacheManager.get_seq_length(past_key_values)
        if target_seq_len >= current_len:
            return past_key_values, (time.perf_counter() - start_time) * 1000.0

        diff = target_seq_len - current_len  # Negative value for crop

        # Modern DynamicCache with .crop or direct layer slicing
        if hasattr(past_key_values, "crop"):
            try:
                past_key_values.crop(diff)
            except Exception:
                # Direct fallback on layer tensors
                if hasattr(past_key_values, "layers"):
                    for layer in past_key_values.layers:
                        if hasattr(layer, "keys") and layer.keys is not None:
                            layer.keys = layer.keys[:, :, :target_seq_len, :].contiguous()
                        if hasattr(layer, "values") and layer.values is not None:
                            layer.values = layer.values[:, :, :target_seq_len, :].contiguous()
            new_cache = past_key_values

        elif isinstance(past_key_values, (tuple, list)):
            new_cache_layers = []
            for layer in past_key_values:
                k, v = layer[0], layer[1]
                k_sliced = k[:, :, :target_seq_len, :].contiguous()
                v_sliced = v[:, :, :target_seq_len, :].contiguous()
                new_cache_layers.append((k_sliced, v_sliced))
            new_cache = tuple(new_cache_layers)
        else:
            raise TypeError(f"Unsupported cache structure: {type(past_key_values)}")

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return new_cache, latency_ms

    @staticmethod
    def compute_cache_memory_mb(past_key_values: Any) -> float:
        """Estimates total GPU memory consumed by past key-values in MB."""
        if past_key_values is None:
            return 0.0
        total_bytes = 0
        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                if hasattr(layer, "keys") and layer.keys is not None:
                    total_bytes += layer.keys.numel() * layer.keys.element_size()
                if hasattr(layer, "values") and layer.values is not None:
                    total_bytes += layer.values.numel() * layer.values.element_size()
        elif isinstance(past_key_values, (tuple, list)):
            for layer in past_key_values:
                k, v = layer[0], layer[1]
                total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total_bytes / (1024.0 * 1024.0)

