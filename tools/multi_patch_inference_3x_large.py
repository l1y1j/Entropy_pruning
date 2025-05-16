import copy
from itertools import product
from math import ceil
from pathlib import Path

# from mmdet.datasets import replace_ImageToTensor

import warnings
import glob
import os
import pickle
import tqdm

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
# from mmcv.parallel import collate, scatter
# from mmdet.datasets import replace_ImageToTensor
# from mmdet.datasets.pipelines import Compose
from mmcv.transforms import Compose
from torch.utils.data import Dataset

# from mmdet.models import build_detector
# from mmcv.runner import load_checkpoint
from mmcv.ops import soft_nms as nms

from argparse import ArgumentParser
from mmdet.utils import register_all_modules
from mmdet.utils import get_test_pipeline_cfg
import torch
import cv2

from ensemble_boxes import weighted_boxes_fusion

import time

all_time = 0
# def nms(bounding_boxes, Nt):
#     if len(bounding_boxes) == 0:
#         return [], []
#     bboxes = np.array(bounding_boxes)
#     x1 = bboxes[:, 0]
#     y1 = bboxes[:, 1]
#     x2 = bboxes[:, 2]
#     y2 = bboxes[:, 3]
#     scores = bboxes[:, 4]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = np.argsort(scores)
#     picked_boxes = []
#     while order.size > 0:
#         index = order[-1]
#         picked_boxes.append(bounding_boxes[index])
#         x11 = np.maximum(x1[index], x1[order[:-1]])
#         y11 = np.maximum(y1[index], y1[order[:-1]])
#         x22 = np.minimum(x2[index], x2[order[:-1]])
#         y22 = np.minimum(y2[index], y2[order[:-1]])
#         w = np.maximum(0.0, x22 - x11 + 1)
#         h = np.maximum(0.0, y22 - y11 + 1)
#         intersection = w * h
#         ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
#         left = np.where(ious < Nt)
#         order = order[left]
#     return picked_boxes



from mmdet.apis import init_detector
# def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
#     """Initialize a detector from config file.
#
#     Args:
#         config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
#             :obj:`Path`, or the config object.
#         checkpoint (str, optional): Checkpoint path. If left as None, the model
#             will not load any weights.
#         cfg_options (dict): Options to override some settings in the used
#             config.
#
#     Returns:
#         nn.Module: The constructed detector.
#     """
#     if isinstance(config, (str, Path)):
#         config = mmcv.Config.fromfile(config)
#     elif not isinstance(config, mmcv.Config):
#         raise TypeError('config must be a filename or Config object, '
#                         f'but got {type(config)}')
#     if cfg_options is not None:
#         config.merge_from_dict(cfg_options)
#     if 'pretrained' in config.model:
#         config.model.pretrained = None
#     elif 'init_cfg' in config.model.backbone:
#         config.model.backbone.init_cfg = None
#     config.model.train_cfg = None
#     model = build_detector(config.model, test_cfg=config.get('test_cfg'))
#     if checkpoint is not None:
#         checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
#         if 'CLASSES' in checkpoint.get('meta', {}):
#             model.CLASSES = checkpoint['meta']['CLASSES']
#         else:
#             warnings.simplefilter('once')
#             warnings.warn('Class names are not saved in the checkpoint\'s '
#                           'meta data, use COCO classes by default.')
#             model.CLASSES = get_classes('coco')
#     model.cfg = config  # save the config in the model for convenience
#     model.to(device)
#     model.eval()
#     return model


def get_multiscale_patch(sizes, steps, ratios):
    """Get multiscale patch sizes and steps.

    Args:
        sizes (list): A list of patch sizes.
        steps (list): A list of steps to slide patches.
        ratios (list): Multiscale ratios. devidie to each size and step and
            generate patches in new scales.

    Returns:
        new_sizes (list): A list of multiscale patch sizes.
        new_steps (list): A list of steps corresponding to new_sizes.
    """
    assert len(sizes) == len(steps), 'The length of `sizes` and `steps`' \
                                     'should be the same.'
    new_sizes, new_steps = [], []
    size_steps = list(zip(sizes, steps))
    for (size, step), ratio in product(size_steps, ratios):
        new_sizes.append(int(size / ratio))
        new_steps.append(int(step / ratio))
    return new_sizes, new_steps

def slide_window(width, height, sizes, steps, img_rate_thr=0.6):
    """Slide windows in images and get window position.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        sizes (list): List of window's sizes.
        steps (list): List of window's steps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        np.ndarray: Information of valid windows.
    """
    assert 1 >= img_rate_thr >= 0, 'The `in_rate_thr` should lie in 0~1'
    windows = []
    # Sliding windows.
    for size, step in zip(sizes, steps):
        size_w, size_h = size
        step_w, step_h = step

        x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
        x_start = [step_w * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size_w > width:
            x_start[-1] = width - size_w

        y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
        y_start = [step_h * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size_h > height:
            y_start[-1] = height - size_h

        start = np.array(list(product(x_start, y_start)), dtype=np.int64)
        windows.append(np.concatenate([start, start + size], axis=1))
    windows = np.concatenate(windows, axis=0)

    # Calculate the rate of image part in each window.
    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates >= img_rate_thr).any():
        img_rates[img_rates == img_rates.max()] = 1
    return windows[img_rates >= img_rate_thr]

def merge_results(results, offsets, iou_thr=0.6, device='cpu'):
    """Merge patch results via nms.

    Args:
        results (list[np.ndarray]): A list of patches results.
        offsets (np.ndarray): Positions of the left top points of patches.
        iou_thr (float): The IoU threshold of NMS.
        device (str): The device to call nms.

    Retunrns:
        list[np.ndarray]: Detection results after merging.
    """
    assert len(results) == offsets.shape[0], 'The `results` should has the ' \
                                             'same length with `offsets`.'
    merged_results = []
    for results_pre_cls in zip(*results):
        tran_dets = []
        for dets, offset in zip(results_pre_cls, offsets):
            dets[:, :2] += offset
            dets[:, 2:4] += offset
            tran_dets.append(dets)
        tran_dets = np.concatenate(tran_dets, axis=0)

        # #************
        merged_results.append(tran_dets)
        # #************

        # global all_time
        # time_start = time.time()
        #
        # if tran_dets.size == 0:
        #     merged_results.append(tran_dets)
        # else:
        #     tran_dets = torch.from_numpy(tran_dets)
        #     tran_dets = tran_dets.to(device)
        #     nms_dets, _ = nms(tran_dets[:, :4].contiguous(), tran_dets[:, -1].contiguous(),
        #                               iou_thr)
        #     merged_results.append(nms_dets.cpu().numpy())
        # all_time += (time.time() - time_start)
    return merged_results


def bbox_overlaps(box1, box2, use_iof=False, order='xyxy'):
    # box1 shape: [N,5]   box2 shape:[N,5]  np.ndarray
    box1=torch.from_numpy(box1).float()
    box2=torch.from_numpy(box2).float()

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    if use_iof:
        union = torch.min(area1[:,None],area2)
    else:
        union = (area1[:,None] + area2 - inter)
    iou = inter / union
    return iou.numpy()

def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out


def inference_detector_by_patches(model,
                                  img,
                                  sizes,
                                  steps,
                                  ratios,
                                  merge_iou_thr,
                                  bs=4):
    """inference patches with the detector.
    Split huge image(s) into patches and inference them with the detector.
    Finally, merge patch results on one huge image by nms.
    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray or): Either an image file or loaded image.
        sizes (list): The sizes of patches.
        steps (list): The steps between two patches.
        ratios (list): Image resizing ratios for multi-scale detecting.
        merge_iou_thr (float): IoU threshold for merging results.
        bs (int): Batch size, must greater than or equal to 1.
    Returns:
        list[np.ndarray]: Detection results.
    """

    # if isinstance(img, (list, tuple)):
    #     is_batch = True
    # else:
    #     img = [img]
    #     is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    cfg = model.cfg

    cfg = cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    if isinstance(img[0], np.ndarray):
        # Calling this method across libraries will result
        # in module unregistered error if not prefixed with mmdet.
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    # # tta
    # if True:
    #     test_data_cfg = test_pipeline
    #     while 'dataset' in test_data_cfg:
    #         test_data_cfg = test_data_cfg['dataset']
    #
    #     # batch_shapes_cfg will force control the size of the output image,
    #     # it is not compatible with tta.
    #     if 'batch_shapes_cfg' in test_data_cfg:
    #         test_data_cfg.batch_shapes_cfg = None
    #     test_data_cfg.pipeline = cfg.tta_pipeline
    #
    # if model.data_preprocessor.device.type == 'cpu':
    #     for m in model.modules():
    #         assert not isinstance(
    #             m, RoIPool
    #         ), 'CPU inference with RoIPool is not supported currently.'


    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    # test_pipeline = Compose(cfg.data.test.pipeline)

    if not isinstance(img, np.ndarray):
        img = mmcv.imread(img)


    results = []

    height, width = img.shape[:2]
    # sizes, steps = get_multiscale_patch(sizes, steps, ratios)
    # windows = slide_window(width, height, sizes, steps)
    # windows = slide_window(width, height, [(4096, 2048)], [(4096-200, 2048-200)])
    # windows = slide_window(width, height, [(4096*3, 2048*2)], [(4096*2, 2048*2)])
    # windows = slide_window(width, height, [(1024 * 3, 1024 * 3)], [(1024 * 2, 1024 * 2)])
    # windows = slide_window(width, height, [(1024 * 3, 1024 * 3)], [(1024 * 2, 1024 * 2)])
    # windows = slide_window(width, height, [(512 * 7, 512 * 7)], [(512 * 3, 512 * 3)])
    # windows = slide_window(width, height, [(512 * 14, 512 * 14)], [(512 * 5, 512 * 5)])
    # windows = slide_window(width, height, [(512 * 10, 512 * 10)], [(512 * 4, 512 * 4)])
    # windows = slide_window(width, height, [(512 * 8, 512 * 8)], [(512 * 5, 512 * 5)])
    # windows = slide_window(width, height, [(6144, 3072)], [(6144-1000, 3072-1000)])
    # windows = slidmue_window(width, height, [(3333, 2000)], [(3333, 2000)])
    # windows = slide_window(width, height, [(2000, 2000), (4000, 4000), (7500, 7500)], [(1000, 1000), (2000, 2000), (3800, 3800)])
    # windows = slide_window(width, height, [(1333*3, 800*3), (width, height)], [(1333*3-500, 800*3-500), (width, height)])
    # windows = slide_window(width, height, [(width//4, height//4)], [(width//8, height//8)])
    # windows = slide_window(width, height, [(width // 4, height // 4), (width // 2, height // 2)], [(width // 5, height // 5), (width // 3, height // 3)])
    # windows = slide_window(width, height, [(1024 * 4, 1024 * 4)], [(1024 * 3, 1024 * 3)])
    windows = slide_window(width, height, [(3000, 3000)], [(1500, 1500)])
    # windows = slide_window(width, height, [(3000, 3000), (6000, 6000), (9000, 9000)], [(1500, 1500), (3000, 3000), (4500, 4500)])
    # windows = slide_window(width, height, [(width // 6, height // 6), (width // 4, height // 4), (width // 2, height // 2)], [(width // 7, height // 7), (width // 5, height // 5), (width // 3, height // 3)])

    # windows = slide_window(width, height, [(2000*3, 1200*3)], [(2000*3-600, 1200*3-600)])
    # windows = slide_window(width, height, [(2048, 1024)], [(2048-200, 1024-200)])

    start = 0

    time_start = time.time()
    while True:
        # prepare patch data
        patch_datas = []
        data_samples_temp = []
        if (start + bs) > len(windows):
            end = len(windows)
        else:
            end = start + bs
        for window in windows[start:end]:
            x_start, y_start, x_stop, y_stop = window
            # patch_width = x_stop - x_start
            # patch_height = y_stop - y_start
            patch = img[y_start:y_stop, x_start:x_stop]
            # prepare data

            data = dict(img=patch, img_id=0)
            data = test_pipeline(data)
            patch_datas.append(data['inputs'])
            data_samples_temp.append(data['data_samples'])

        data['inputs'] = patch_datas
        data['data_samples'] = data_samples_temp

        # # just get the actual data from DataContainer
        # data['img_metas'] = [
        #     img_metas.data[0] for img_metas in data['img_metas']
        # ]
        # data['img'] = [img.data[0] for img in data['img']]
        # if next(model.parameters()).is_cuda:
        #     # scatter to specified GPU
        #     data = scatter(data, [device])[0]
        # else:
        #     for m in model.modules():
        #         assert not isinstance(
        #             m, RoIPool
        #         ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            results_temp = model.test_step(data)
            results_temp = [[torch.cat([result.pred_instances.bboxes, result.pred_instances.scores.unsqueeze(1)],
                                  dim=1).cpu().numpy()] for result in results_temp]
            # results.extend(results_temp)

            results_temp_filter = []
            for result, window in zip(results_temp, windows):
                result = result[0]
                x_start, y_start, x_stop, y_stop = window
                temp = []
                p_width = x_stop - x_start
                p_height = y_stop - y_start
                x_margin = p_width * 0.05
                y_margin = p_height * 0.05
                # print(x_margin, y_margin)
                patch = copy.deepcopy(patch)
                for i in range(result.shape[0]):
                    x1, y1, x2, y2, s = result[i]
                    area = (x2 - x1) * (y2 - y1)
                    if x1 < x_margin or x2 > p_width-x_margin or y1 < y_margin or y2 > p_height-y_margin:
                        # print(x1, x2, y1, y2)
                        continue
                    # if s > 0.3:
                    #     x1, y1, x2, y2, _ = map(int, result[i])
                        # print(x1, y1,x2, y2 )
                        # patch = cv2.rectangle(patch, (x1, y1), (x2, y2), (255, 0, 0), 10)
                    temp.append(result[i])
                    # if area < 700 * 700:
                    #     temp.append(result[i])

                # import uuid
                # id = uuid.uuid4()
                # print(id)
                # print("*"*100)
                # cv2.imwrite("patch_"+str(id) + '.jpg', patch)


                if len(temp)==0:
                    temp = np.zeros((0, 5), np.float32)
                else:
                    temp = np.array(temp)
                results_temp_filter.append([temp])
            results.extend(results_temp_filter)

        if end >= len(windows):
            break
        start += bs


    windows_trick = slide_window(width, height, [(width, height)], [(width, height)])
    patch_datas = []
    data_samples_temp = []
    for window in windows_trick:
        x_start, y_start, x_stop, y_stop = window
        # patch_width = x_stop - x_start
        # patch_height = y_stop - y_start
        patch = img[y_start:y_stop, x_start:x_stop]
        # prepare data

        data = dict(img=patch, img_id=0)
        data = test_pipeline(data)
        patch_datas.append(data['inputs'])
        data_samples_temp.append(data['data_samples'])

    data['inputs'] = patch_datas
    data['data_samples'] = data_samples_temp

    # forward the model
    with torch.no_grad():
        results_temp = model.test_step(data)
        results_temp = [[torch.cat([result.pred_instances.bboxes, result.pred_instances.scores.unsqueeze(1)],
                                   dim=1).cpu().numpy()] for result in results_temp]

        results_temp_filter = []
        for result in results_temp:
            result = result[0]
            temp = []
            for i in range(result.shape[0]):
                x1, y1, x2, y2, _ = result[i]
                area = (x2-x1) * (y2-y1)
                if area > 700 * 700:
                    temp.append(result[i])

            if len(temp) == 0:
                temp = np.zeros((0, 5), np.float32)
            else:
                temp = np.array(temp)
            results_temp_filter.append([temp])
        results.extend(results_temp_filter)


    global all_time
    all_time += (time.time()-time_start)
    print(time.time()-time_start)
    # print(time.time()-time_start)
    # print(windows.shape, windows_trick.shape)
    results = merge_results(
        results,
        np.concatenate((windows, windows_trick), axis=0)[:, :2],
        # windows[:, :2],
        iou_thr=merge_iou_thr,
        device=device)
    return results

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img_path', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.5,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

class PANDA(Dataset):
    def __init__(self, mode="train", **kwargs):
        self.root = "/home/wenxi/panda/images" if mode == "train" else \
                "/data/liwenxi/panda/images_test"
        temp = []
        self.paths = glob.glob(os.path.join(self.root, '*jpg'))
        self.paths.sort()
        self.gt_type = kwargs['gt_type']
        if mode == "train":
            for path in self.paths:
                name = os.path.basename(path)
                tag = name.split('.')[-2].split('_')[-1]
                if tag not in ['01', '06', '11', '16', '21', '26']:
                    temp.append(path)
        else:
            for path in self.paths:
                name = os.path.basename(path)
                tag = name.split('.')[-2].split('_')[-1]
                temp.append(path)
        self.paths = temp
        # self.paths = self.paths[:4]
        self.transform = kwargs['transform']
        self.length = len(self.paths)
        self.load_raw_img = kwargs['raw']
        # self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.load_raw_img:
            img_path = self.paths[item]
            raw_path = img_path
            # raw_path = img_path
            raw = cv2.imread(raw_path)
            name = os.path.basename(img_path)
        # img, den = self.load_data(item)
        img, den = torch.rand(1), torch.rand(1)
        if self.load_raw_img:
            return img, den, raw, name
        return img, den


def main(args):
    register_all_modules()
    all_result = []
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a huge image by patches

    root = "/media/wzh/wxli/PANDA/images_test"
    # root = "/data3/wxli/panda/images_test"

    paths = glob.glob(os.path.join(root, '*jpg'))
    paths.sort()

    from torchvision import transforms
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset_test = PANDA(mode="test", transform=transform, raw=True, gt_type='adaptive_4scale_16')
    # print(dataset_test.__len__())
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)


    # for img in tqdm.tqdm(paths):
    for img, density, raw, name in tqdm.tqdm(dataloader_test):
        img = raw.squeeze().numpy()
        result = inference_detector_by_patches(model, img, args.patch_sizes,
                                               args.patch_steps, args.img_ratios,
                                               args.merge_iou_thr)


        # for box in result[0]:
        #     x1, y1, x2, y2 = map(int, box[:4])
        #     s = box[4]
        #     if s < 0.2:
        #         continue
        #     img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
        # import uuid
        # print("*"*100)
        # print(name)
        # cv2.imwrite(name[0], img)
        # # cv2.imwrite(str(uuid.uuid4())+'.jpg', img)

        # print(result)
        all_result.append(result)
    print(all_time/len(all_result))
    with open('./large_wo_NMS.pkl', 'wb') as f:
        pickle.dump(all_result, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)

    device = 'cuda:0'
    # device = 'cpu'
    all_result = []
    with open('large_wo_NMS.pkl', 'rb') as f:
        outputs = pickle.load(f)

    for results in outputs:
        merged_results = []
        for tran_dets in results:
            tran_dets = torch.from_numpy(tran_dets)
            tran_dets = tran_dets.to(device)
            nms_dets, _ = nms(tran_dets[:, :4].contiguous(), tran_dets[:, -1].contiguous(),
                                      iou_thr=0.7)
            # w = torch.max(tran_dets[:, 2]).cpu().numpy()
            # h = torch.max(tran_dets[:, 3]).cpu().numpy()
            # boxes_list = [nms_dets.cpu().numpy()[:, :4]/w, tran_dets.cpu().numpy()[:, :4]/w]
            # scores_list = [nms_dets.cpu().numpy()[:, 4], tran_dets.cpu().numpy()[:, 4]]
            # labels_list = [np.ones(scores_list[0].shape), np.ones(scores_list[1].shape)]
            # weights = [2, 1]
            # iou_thr = 0.5
            # skip_box_thr = 0.0001
            # sigma = 0.1
            # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            # vote_out = box_voting(nms_dets.cpu().numpy(), tran_dets.cpu().numpy(), thresh=0.8, scoring_method='TEMP_AVG')

            merged_results.append(nms_dets.cpu().numpy()[:1000])
            # merged_results.append(vote_out[:1000])
            # merged_results.append(np.concatenate((boxes[:1000]*w, scores[:1000, np.newaxis]), axis=1))
            # break
        # print(nms_dets.shape)
        # all_result.append([nms_dets.cpu().numpy()])
        all_result.append(merged_results)


    # for results in outputs:
    #     merged_results = []
    #     for tran_dets in results:
    #         tran_dets = torch.from_numpy(tran_dets)
    #         # tran_dets = tran_dets.to(device)
    #         w = float(torch.max(tran_dets[:, 2]))
    #         h = float(torch.max(tran_dets[:, 3]))
    #         tran_dets[:, 0] = tran_dets[:, 0] / w
    #         tran_dets[:, 1] = tran_dets[:, 1] / h
    #         tran_dets[:, 2] = tran_dets[:, 2] / w
    #         tran_dets[:, 3] = tran_dets[:, 3] / h
    #
    #         weights = [2, 1]
    #         iou_thr = 0.5
    #         skip_box_thr = 0.0001
    #         sigma = 0.1
    #
    #         # nms_dets, scores, _ = weighted_boxes_fusion([tran_dets[:, :4].contiguous().tolist()], [tran_dets[:, -1].contiguous().tolist()], [torch.zeros(tran_dets[:, -1].shape).tolist()], weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    #
    #         nms_dets[:, 0] = nms_dets[:, 0] * w
    #         nms_dets[:, 1] = nms_dets[:, 1] * h
    #         nms_dets[:, 2] = nms_dets[:, 2] * w
    #         nms_dets[:, 3] = nms_dets[:, 3] * h
    #         nms_dets = np.concatenate((nms_dets, scores[:, np.newaxis]), axis=1)
    #         merged_results.append(nms_dets[:1000])
    #         # break
    #     # print(nms_dets.shape)
    #     # all_result.append([nms_dets.cpu().numpy()])
    #     all_result.append(merged_results)


    # print(all_result)
    with open('large_filter_kuai.pkl', 'wb') as f:
        pickle.dump(all_result, f)