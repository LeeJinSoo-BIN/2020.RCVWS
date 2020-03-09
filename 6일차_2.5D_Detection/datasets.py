import torch
import torch.utils.data as data
import json
import os
import os.path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import *
from collections import namedtuple
from torchcv.datasets import UnNormalize, Compose, ToTensor, ToPILImage, Normalize, Resize, RandomHorizontalFlip, RandomResizedCrop, ColorJitter

DB_ROOT = './datasets/New_Sejong_RCV_dataset'

OBJ_LOAD_CONDITIONS = {    
    'train': {'hRng': (-np.inf, np.inf), 'xRng':(-np.inf, np.inf), 'yRng':(-np.inf, np.inf), 'wRng':(-np.inf, np.inf)},
    'test': {'hRng': (-np.inf, np.inf), 'xRng':(-np.inf, np.inf), 'yRng':(-np.inf, np.inf), 'wRng':(-np.inf, np.inf)}
}


class Sejong_Ped(data.Dataset):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """

    def __init__(self,image_set,img_transform=None, co_transform=None, condition=None):

        assert condition in OBJ_LOAD_CONDITIONS
        
        self.img_transform = img_transform
        self.co_transform = co_transform        
        self.cond = OBJ_LOAD_CONDITIONS[condition]

        self._parser = LoadBox()    

        self._annopath_New_Sejong = os.path.join('%s','RGBTDv3/json_depth_label_4','%s','%s.json')
        self._imgpath_vis_New_Sejong = os.path.join('%s','RGBTDv3/Image','%s','RGB','I%s.png')
        self._imgpath_lwir_New_Sejong = os.path.join('%s','RGBTDv3/Image','%s','Ther','I%s.png') 

        self.ids = list() 

        for line in open(os.path.join(DB_ROOT, 'RGBTDv3/ImageSet', image_set)):
            self.ids.append((DB_ROOT, line.strip().split('/')))


    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set

    def __getitem__(self, index): 

        vis, lwir, boxes, labels, depths = self.pull_item(index)
        return vis, lwir, boxes, labels, depths, torch.ones(1,dtype=torch.int)*index

    def pull_item(self, index):

        frame_id = self.ids[index]

        with open(self._annopath_New_Sejong %(DB_ROOT,frame_id[1][0],frame_id[1][1])) as j:
            target = json.load(j)
            
        # isDay = DAY_NIGHT_CLS[set_id]       
        
        vis = Image.open(self._imgpath_vis_New_Sejong%(DB_ROOT,frame_id[1][0],frame_id[1][1][-7:]))
        lwir = Image.open(self._imgpath_lwir_New_Sejong%(DB_ROOT,frame_id[1][0],frame_id[1][1][-7:])).convert('L')

        width, height = vis.size

        boxes = self._parser(target, width, height)
        
        depths = boxes[:,5]
        boxes = boxes[:,:5]

        ## Apply transforms
        if self.img_transform is not None:
            vis,lwir,_ = self.img_transform(vis,lwir)


        if self.co_transform is not None:                    
            vis,lwir,boxes = self.co_transform(vis,lwir,box=boxes)


        # import pdb;pdb.set_trace()
        # vis = np.array( tensor2image( vis.clone() ) [0])
        # vis = Image.fromarray(vis, 'RGB')
        # vis.save('vis_after.png')
        # boxes[:,2] = (boxes[:,2]-boxes[:,0])*width
        # boxes[:,3] = (boxes[:,3]-boxes[:,1])*height
        # boxes[:,0] = boxes[:,0]*width
        # boxes[:,1] = boxes[:,1]*height
        

        ## Set ignore flags
        ignore = torch.zeros( boxes.size(0), dtype=torch.uint8)
               
        for ii, box in enumerate(boxes):
                        
            x = box[0] * width
            y = box[1] * height
            w = ( box[2] - box[0] ) * width
            h = ( box[3] - box[1] ) * height

            if  x < self.cond['xRng'][0] or \
                y < self.cond['xRng'][0] or \
                x+w > self.cond['xRng'][1] or \
                y+h > self.cond['xRng'][1] or \
                w < self.cond['wRng'][0] or \
                w > self.cond['wRng'][1] or \
                h < self.cond['hRng'][0] or \
                h > self.cond['hRng'][1]:

                ignore[ii] = 1
        
        boxes[ignore, 4] = -1

        labels = boxes[:,4]
        boxes_t = boxes[:,0:4]

        return vis, lwir, boxes_t, labels, depths


    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        vis = list()
        lwir = list()
        boxes = list()
        labels = list()
        index = list()
        depths = list()

        for b in batch:
            vis.append(b[0])
            lwir.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            depths.append(b[4])
            index.append(b[5])

        vis = torch.stack(vis, dim=0)
        lwir = torch.stack(lwir, dim=0)
   
        return vis, lwir, boxes, labels, depths, index  

class LoadBox(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, bbs_format='xyxy'):

        assert bbs_format in ['xyxy', 'xywh']                
        self.bbs_format = bbs_format
        self.pts = ['x', 'y', 'w', 'h']

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation 
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """                

        res = [ [0, 0, 0, 0, -1, 0] ]

        for ii in range(len(target["annotation"])):

            occlusion = target["annotation"][ii]["occlusion"]
            is_crowd = target["annotation"][ii]["iscrowd"]
            bndbox = target["annotation"][ii]["bbox"]
            label_idx = target["annotation"][ii]["category_id"]
            depth = target["annotation"][ii]["depth"]/1000

            if depth < 0 :
                depth = 0
                label_idx = -1

            if occlusion == 2 :
                continue

            bndbox[0] = min(bndbox[0], width)
            bndbox[1] = min(bndbox[1], height)
            bndbox[2] = min(bndbox[2], width)
            bndbox[3] = min(bndbox[3], height)

            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]

            bndbox.append(label_idx)
            bndbox.append(depth)           

            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return np.array(res, dtype=np.float)  # [[xmin, ymin, xmax, ymax, label_ind]]