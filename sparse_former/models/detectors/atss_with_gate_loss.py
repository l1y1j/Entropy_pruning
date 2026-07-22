import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.atss import ATSS


@MODELS.register_module()
class ATSSWithGateLoss(ATSS):
    """ATSS detector with additional gate loss from SwinTransformerV3 backbone."""

    def loss(self, batch_inputs, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)

        # Collect gate loss from backbone (same pattern as DINOWithGateLoss).
        # Two patterns supported:
        #   (A) dict-based (SparseNetEntropy):  backbone.get_gate_losses() -> dict
        #   (B) buffer-based (SwinTransformerV3):  backbone.kl_predictor_gate_loss etc.
        if hasattr(self.backbone, 'get_gate_losses'):
            gate_losses = self.backbone.get_gate_losses()
            if gate_losses:
                losses.update(gate_losses)
        else:
            if hasattr(self.backbone, 'kl_predictor_gate_loss'):
                kl_loss = self.backbone.kl_predictor_gate_loss
                if kl_loss is not None and isinstance(kl_loss, torch.Tensor):
                    losses['loss_kl_gate'] = kl_loss
            if hasattr(self.backbone, 'kl_predictor_gate_info'):
                kl_info = self.backbone.kl_predictor_gate_info
                if kl_info is not None and isinstance(kl_info, torch.Tensor):
                    losses['loss_kl_gate_info'] = kl_info
            if hasattr(self.backbone, 'kl_predictor_gate_reg'):
                kl_reg = self.backbone.kl_predictor_gate_reg
                if kl_reg is not None and isinstance(kl_reg, torch.Tensor):
                    losses['loss_kl_gate_reg'] = kl_reg
            if hasattr(self.backbone, 'inc_predictor_gate_loss'):
                inc_loss = self.backbone.inc_predictor_gate_loss
                if inc_loss is not None and isinstance(inc_loss, torch.Tensor):
                    losses['loss_inc_gate'] = inc_loss
            if hasattr(self.backbone, 'inc_predictor_gate_info'):
                inc_info = self.backbone.inc_predictor_gate_info
                if inc_info is not None and isinstance(inc_info, torch.Tensor):
                    losses['loss_inc_gate_info'] = inc_info
            if hasattr(self.backbone, 'inc_predictor_gate_reg'):
                inc_reg = self.backbone.inc_predictor_gate_reg
                if inc_reg is not None and isinstance(inc_reg, torch.Tensor):
                    losses['loss_inc_gate_reg'] = inc_reg

        # Hard prune rate (no loss_ prefix, logging metric only)
        if hasattr(self.backbone, 'kl_predictor_hard_total'):
            total_win = self.backbone.kl_predictor_hard_total
            kl_keep = self.backbone.kl_predictor_hard_kl_keep
            final_keep = self.backbone.kl_predictor_hard_final_keep
            if total_win > 0:
                losses['kl_prune_rate'] = torch.tensor(1.0 - kl_keep / total_win)
                losses['final_prune_rate'] = torch.tensor(1.0 - final_keep / total_win)
                if kl_keep > 0:
                    losses['inc_prune_rate'] = torch.tensor(1.0 - final_keep / kl_keep)

        if hasattr(self.backbone, 'kl_predictor_avg_tau'):
            losses['kl_avg_tau'] = torch.tensor(self.backbone.kl_predictor_avg_tau)
        if hasattr(self.backbone, 'inc_predictor_avg_tau'):
            losses['inc_avg_tau'] = torch.tensor(self.backbone.inc_predictor_avg_tau)

        return losses
