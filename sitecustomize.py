try:
    from transformers.utils import import_utils as _transformers_import_utils

    mapping = getattr(_transformers_import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if isinstance(mapping, dict):
        mapping.setdefault("flash_attn", ("flash-attn", "flash_attn"))
except Exception:
    pass
