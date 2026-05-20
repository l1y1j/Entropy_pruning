import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.dino import DINO


@MODELS.register_module()
class DINOWithGateLoss(DINO):
    """DINO detector with additional gate loss from SwinTransformerV2 backbone."""

    def loss(self, batch_inputs, batch_data_samples):
        img_feats = self.extract_feat(batch_inputs)
        
        # 数值检查：确保 backbone 输出的特征不包含 NaN/Inf
        # 使用极小的随机噪声代替 0.0，防止 LayerNorm 方差为 0 导致新的 Inf
        for i, feat in enumerate(img_feats):
            if not torch.isfinite(feat).all():
                noise = torch.randn_like(feat) * 1e-4
                img_feats[i] = torch.where(torch.isfinite(feat), feat, noise)
        
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        
        # 数值检查：确保输入到检测头的特征不包含 NaN/Inf
        for k, v in head_inputs_dict.items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                noise = torch.randn_like(v) * 1e-4
                head_inputs_dict[k] = torch.where(torch.isfinite(v), v, noise)
        
        losses = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples)

        # Add gate loss from SwinTransformerV3 backbone (V2 uses _kl_gate_loss/_inc_gate_loss)
        if hasattr(self.backbone, 'kl_predictor_gate_loss'):
            kl_loss = self.backbone.kl_predictor_gate_loss
            if kl_loss is not None and isinstance(kl_loss, torch.Tensor):
                losses['loss_kl_gate'] = kl_loss
        if hasattr(self.backbone, 'inc_predictor_gate_loss'):
            inc_loss = self.backbone.inc_predictor_gate_loss
            if inc_loss is not None and isinstance(inc_loss, torch.Tensor):
                losses['loss_inc_gate'] = inc_loss

        return losses