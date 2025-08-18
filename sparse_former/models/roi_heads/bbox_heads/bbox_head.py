from mmdet.models import BBoxHead as MM_BBoxHead
from mmengine.config import ConfigDict
from sparse_former.registry import MODELS

@MODELS.register_module()
class BBoxHead(MM_BBoxHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        in_channels = self.in_channels
        if not self.with_avg_pool:
            in_channels *= self.roi_feat_area
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=in_channels, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)