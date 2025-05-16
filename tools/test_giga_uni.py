import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab
import tqdm
import pickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    annFile = './p_test.json'
    detFile = '/home/liwenxi/mmdetection/output_4000x4000_trt.pkl'
    root = "/data/liwenxi/panda/images_test"

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here

    # initialize COCO ground truth api
    cocoGt = COCO(annFile)

    paths = glob.glob(os.path.join(root, '*jpg'))
    paths.sort()

    name_id_map = {}
    for id, name in enumerate(paths):
        short_name = os.path.basename(name)
        name_id_map[short_name] = id + 1

    coco_result = []

    with open(detFile, 'rb') as f:
        outputs = pickle.load(f)

    for img, bbox in zip(paths, outputs):
        name = os.path.basename(img)
        for item in bbox[0]:
            item = item.astype(np.float64)
            box = {}
            box['image_id'] = name_id_map[name]
            box['category_id'] = 1
            box['bbox'] = [item[0], item[1], item[2] - item[0], item[3] - item[1]]
            box['score'] = item[4]
            coco_result.append((name_id_map[name], item[0], item[1], item[2] - item[0], item[3] - item[1], item[4], 1))

    save_data = np.array(coco_result)

    # initialize COCO detections api
    cocoDt = cocoGt.loadRes(save_data)

    imgIds = sorted(cocoGt.getImgIds())

    # # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [500, 500, 100]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
