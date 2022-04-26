from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, gt_root, trimap_root):
        if os.path.isdir(pred_root):
            pred_dirs = os.listdir(pred_root)
        else:
            raise Exception(pred_root + ' not dir')

        if os.path.isdir(gt_root):
            gt_dirs = os.listdir(gt_root)
        else:
            raise Exception(gt_root +' not dir')

        if os.path.isdir(trimap_root):
            trimap_dirs = os.listdir(trimap_root)
        else:
            raise Exception(trimap_root +' not dir')

        dir_name_list = []
        for idir in pred_dirs:
            if idir in gt_dirs and idir in trimap_dirs:
                dir_name_list.append(idir)
                        

        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), dir_name_list))
        self.gt_path = list(
            map(lambda x: os.path.join(gt_root, x), dir_name_list))
        self.trimap_path = list(
            map(lambda x: os.path.join(trimap_root, x), dir_name_list))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.gt_path[item]).convert('L')
        trimap = Image.open(self.trimap_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        if trimap.size != gt.size:
            trimap = trimap.resize(gt.size, Image.BILINEAR)

        return pred, gt, trimap

    def __len__(self):
        return len(self.image_path)
