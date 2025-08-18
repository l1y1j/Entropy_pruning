from mmdet.models import ConvFCBBoxHead as MM_ConvFCBBoxHead
from mmengine.config import ConfigDict
from sparse_former.registry import MODELS


@MODELS.register_module()
class ConvFCBBoxHead(MM_ConvFCBBoxHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)