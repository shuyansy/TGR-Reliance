"""Optional (--hf_mrope): make vLLM 0.11 compute Qwen2.5-VL M-RoPE time positions exactly as transformers 4.57.

HF:   second_per_grid_t is truncated to an integer (int64), then t * k * tokens_per_second in int64.
vLLM: second_per_grid_t is a bf16 tensor and the product is computed in bf16, so positions above 256 can be rounded even when
      k is an exact integer (long videos, e.g. 3,665 s at 128 frames: up to 16 positions off).
The patch passes Python ints to MRotaryEmbedding.get_input_positions_tensor for qwen2_5_vl. It only works when the vLLM engine
runs in the calling process (VLLM_ENABLE_V1_MULTIPROCESSING=0, set by eval_oasis.py).
Note: the Video-Oasis numbers of our trained models in the paper were computed without this patch.
"""
import torch

_PATCHED = False


def apply():
    global _PATCHED
    if _PATCHED:
        return
    from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
    orig = MRotaryEmbedding.get_input_positions_tensor.__func__

    def patched(cls, input_tokens, *args, **kwargs):
        hf_config = kwargs.get("hf_config", args[0] if args else None)
        sgt = kwargs.get("second_per_grid_ts")
        if getattr(hf_config, "model_type", "") == "qwen2_5_vl" and sgt:
            kwargs["second_per_grid_ts"] = [int(torch.as_tensor(v).float().reshape(-1)[0].item()) for v in sgt]
        return orig(cls, input_tokens, *args, **kwargs)

    patched.__hf_int_mrope__ = True
    MRotaryEmbedding.get_input_positions_tensor = classmethod(patched)
    _PATCHED = True
