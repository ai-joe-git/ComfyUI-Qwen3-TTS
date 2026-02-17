def patch_rope_default_if_missing():
    """
    Transformers 5.x removed ROPE_INIT_FUNCTIONS['default'].
    Qwen3-TTS expects a 'default' rope type.
    This patch restores compatibility while being a no-op on transformers 4.x.
    """
    try:
        import torch
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" in ROPE_INIT_FUNCTIONS:
            return

        def _rope_init_default(config, device=None, seq_len=None):
            # Determine head dimension
            if hasattr(config, "head_dim") and config.head_dim is not None:
                dim = int(config.head_dim)
            else:
                dim = int(config.hidden_size // config.num_attention_heads)

            base = float(getattr(config, "rope_theta", 10000.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            attention_scaling = 1.0
            return inv_freq, attention_scaling

        ROPE_INIT_FUNCTIONS["default"] = _rope_init_default

    except Exception:
        # If anything goes wrong, don't block startup; the model load will surface the error.
        return
