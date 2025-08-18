"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-23
@desc:   
"""
from .coco import CocoDataset
from sparse_former.registry import DATASETS


@DATASETS.register_module()
class PandaDataset(CocoDataset):
    """ PANDA dataset. """
    pass