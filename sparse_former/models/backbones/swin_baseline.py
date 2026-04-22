from mmdet.registry import MODELS
from mmdet.models.backbones.swin import SwinTransformer


@MODELS.register_module()
class SwinTransformerBaseline(SwinTransformer):
    """Baseline Swin Transformer - No entropy pruning

    This is a pure Swin Transformer backbone without any entropy-based
    window pruning logic. Used for baseline comparison.
    """
    pass