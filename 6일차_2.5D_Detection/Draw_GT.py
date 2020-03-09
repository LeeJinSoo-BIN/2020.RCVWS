from utils import *
from datasets import *
from torchcv.datasets.transforms import *
import torch.nn.functional as F
from tqdm import tqdm
from pprint import PrettyPrinter

import torch
import torch.utils.data as data
import json
import os
import os.path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from utils import *
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import pdb
from collections import namedtuple

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import argparse

from model import SSD300, MultiBoxLoss
from datasets import *
from utils import *
from torchcv.datasets.transforms import *
from torchcv.utils import run_tensorboard
from tensorboardX import SummaryWriter
import numpy as np

##################################################

from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import os
import os.path
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = [512, 640]
batch_size = 1
workers = 0

# Load model checkpoint
checkpoint = './jobs/2020-02-24_23h08m_train_SF+_depth_regression_L1/checkpoint_ssd300.pth.tar022'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.' % (start_epoch))
model = checkpoint['model']
model = model.to(device)
model.eval()

preprocess1 = Compose([ ])    
transforms1 = Compose([ Resize(input_size), \
                        ToTensor(), \
                        Normalize( [0.5873,0.5328,0.4877], [0.2331,0.2160,0.2010], 'R'), \
                        Normalize( [0.4126], [0.1453], 'T')])
                        
test_dataset = Sejong_Ped('test_all.txt',img_transform=preprocess1, co_transform=transforms1, condition='test')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=workers,
                                            collate_fn=test_dataset.collate_fn, 
                                            pin_memory=True)     

resize = transforms.Resize((512, 640))
to_tensor = transforms.ToTensor()
normalize_vis = transforms.Normalize(mean=[0.5873, 0.5328, 0.4877],std=[0.2331, 0.2160, 0.2010])
normalize_lwir = transforms.Normalize(mean=[0.4126],std=[0.1453])
ori_size_out = [422,538]

#Data load
DB_ROOT = './datasets/New_Sejong_RCV_dataset'
image_set = 'test_all.txt'

imgpath_vis_New_Sejong = os.path.join('%s','RGBTDv3/Image','%s','RGB','I%s.png')
imgpath_lwir_New_Sejong = os.path.join('%s','RGBTDv3/Image','%s','Ther','I%s.png') 

ids = list() 

for line in open(os.path.join(DB_ROOT, 'RGBTDv3/ImageSet', image_set)):
    ids.append((DB_ROOT, line.strip().split('/')))


def detect(original_vis, original_lwir, boxes, true_labels, true_depths, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    true_bboxes_out = boxes
    true_depths_out = true_depths[0]

    # Transform
    re_vis = resize(original_vis)
    re_lwir = resize(original_lwir)

    image_vis = normalize_vis(to_tensor(re_vis)).to(device)
    image_lwir = normalize_lwir(to_tensor(re_lwir)).to(device)

    # Forward prop.
    predicted_locs, predicted_scores, predicted_depths = model(image_vis.unsqueeze(0),image_lwir.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores, det_depths = model.detect_objects(predicted_locs, predicted_scores, predicted_depths, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)  
    
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_depths = det_depths[0].to('cpu')
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # Transform to original image dimensions
    original_dims = torch.FloatTensor([original_vis.width, original_vis.height, original_vis.width, original_vis.height]).unsqueeze(0)

    # Annotate
    annotated_image = original_vis
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    det_boxes = det_boxes * original_dims
    true_bboxes_out = true_bboxes_out * original_dims

    for i in range(true_bboxes_out.size(0)):
        # Boxes
        if true_labels[0][i] < 0 :
            continue

        box_location_true = true_bboxes_out[i].tolist()
        true_label_map = rev_label_map[true_labels[0][i].item()]
        draw.rectangle(xy=box_location_true, outline='#FFFFFF')
        draw.rectangle(xy=[l + 1. for l in box_location_true], outline='#FFFFFF')

        text_true_size = font.getsize('GT_'+true_label_map)
        text_true_location = [box_location_true[0] + 2., box_location_true[1] - text_true_size[1]]
        textbox_true_location = [box_location_true[0], box_location_true[1] - text_true_size[1], box_location_true[0] + text_true_size[0] + 4.,box_location_true[1]]
        draw.rectangle(xy=textbox_true_location, fill=label_color_map[true_label_map])
        draw.text(xy=text_true_location, text='{:s}'.format('GT_'+true_label_map), fill='white', font=font)

        cw = (box_location_true[2] - box_location_true[0])/2
        ch = (box_location_true[3] - box_location_true[1])/2

        text_true_depth_size = font.getsize('GT_'+str(round(true_depths_out[i],2)))
        text_true_depth_location = [box_location_true[0] + cw, box_location_true[1] - text_true_depth_size[1] + ch]
        textbox_true_depth_location = [box_location_true[0] + cw, box_location_true[1] - text_true_depth_size[1] + ch, box_location_true[0] + text_true_depth_size[0] + cw,box_location_true[1]+ch]
        draw.rectangle(xy=textbox_true_depth_location, fill=label_color_map[true_label_map])
        draw.text(xy=text_true_depth_location, text='{:s}'.format('GT_'+str(round(true_depths_out[i],2))), fill='white', font=font)
        

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background'] :
        return annotated_image

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])  
        # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_score = det_labels[i]
        text_size = font.getsize('Pred_'+text_score)
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        # draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
        draw.text(xy=text_location, text='{:s}'.format('Pred_'+text_score), fill='white', font=font)

        # Depth Text 
        text_depths = round(det_depths[i].item(),2)
        text_depths_size = font.getsize('Pred_'+str(text_depths))

        pred_cw = (box_location[2] - box_location[0])/2
        pred_ch = (box_location[3] - box_location[1])/2

        text_depth_location = [box_location[0] + pred_cw, box_location[1] - text_depths_size[1] + pred_ch - 30]
        textbox_depth_location = [box_location[0] + pred_cw, box_location[1] - text_depths_size[1] + pred_ch - 30, box_location[0] + text_depths_size[0] + pred_cw,box_location[1]+pred_ch- 30]
        draw.rectangle(xy=textbox_depth_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_depth_location, text='{:s}'.format('Pred_'+str(text_depths)), fill='white', font=font)


        
    # del draw
    
    return annotated_image


if __name__ == '__main__':

    img_id = 0

    with torch.no_grad():
        # Batches
        for (ii, (image_vis, image_lwir, boxes, labels, depths, index)) in tqdm(zip(ids, test_loader), desc='Drawing'):

            frame_id = ii[1]

            ori_vis = resize(Image.open(imgpath_vis_New_Sejong%(DB_ROOT,frame_id[0],frame_id[1][-7:])))
            ori_lwir = resize(Image.open(imgpath_lwir_New_Sejong%(DB_ROOT,frame_id[0],frame_id[1][-7:])))

            annotate_lwir = detect(ori_vis, ori_lwir, boxes[0], labels, depths,  min_score=0.6, max_overlap=0.45, top_k=200)

            annotate_lwir.save('./Detection_visualization_depth/{:06d}.jpg'.format(img_id))
            img_id = img_id+1
  