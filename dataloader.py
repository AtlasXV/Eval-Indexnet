from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        if os.path.isdir(pred_root):
            pred_dirs = os.listdir(pred_root)
        else:
            raise Exception(pred_root + ' not dir')
        if os.path.isdir(label_root):
            label_dirs = os.listdir(label_root)
        else:
            raise Exception(label_root +' not dir')
        dir_name_list = []
        for idir in pred_dirs:
            if idir in label_dirs:
                dir_name_list.append(idir)
                        

        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), dir_name_list))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), dir_name_list))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
