from utils import *
from datasets import *
from torchcv.datasets.transforms import *
import torch.nn.functional as F
from tqdm import tqdm
from pprint import PrettyPrinter
import argparse

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


from torchcv.utils import Timer, kaist_results_file as write_result, write_coco_format as write_result_coco

### Evaluation
from torchcv.evaluations.coco import COCO
from torchcv.evaluations.eval_MR_multisetup import COCOeval

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--synth_fail',         default=['None', 'None'], nargs='+', type=str, help='Specify synthetic failure: e.g. crack1.jpg None')

annType = 'bbox'

DB_ROOT = './datasets/New_Sejong_RCV_dataset/RGBTDv3'
JSON_GT_FILE = os.path.join( DB_ROOT, 'Potenit_ALL_Hard.json' )
cocoGt = COCO(JSON_GT_FILE)

# Parameters
batch_size = 4
workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './jobs/2020-02-11_17h20m_train_SF+_POTENIT_Hard_OCc/checkpoint_ssd300.pth.tar026'
checkpoint_root = './train_SF+_POTENIT'
input_size = [512, 640]
ori_size = (422, 538)
ori_size_out = [422,538]

args = parser.parse_args()

if not args.synth_fail == ['None', 'None']:
    if(str(args.synth_fail[0])=='None') :
        str1 = "None"
    elif(str(args.synth_fail[0])=="blackout") :
        str1 = "blackout"
    else :
        str1 = str(args.synth_fail[0][:-4])

    if(str(args.synth_fail[1])=='None') :
        str2 = "None"
    elif(str(args.synth_fail[1])=="blackout") :
        str2 = "blackout"
    else :
        str2 = str(args.synth_fail[1][:-4])

    checkpoint_name = str1 +'_'+ str2
else :
    checkpoint_name = "Base"

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
preprocess1 = Compose([ ])

if not args.synth_fail == ['None', 'None']:
    fail_mask = [ os.path.join( 'synthetic_failure_masks', mask ) for mask in args.synth_fail ]
    preprocess1.add( [ SynthFail(fail_mask, (ori_size)) ] )

transforms1 = Compose([ Resize(input_size), \
                        ToTensor(), \
                        Normalize( [0.5873,0.5328,0.4877], [0.2331,0.2160,0.2010], 'R'), \
                        Normalize( [0.4126], [0.1453], 'T')])

test_dataset = Sejong_Ped('test_all.txt',img_transform=preprocess1, co_transform=transforms1, condition='test')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=workers,
                                            collate_fn=test_dataset.collate_fn, 
                                            pin_memory=True)     

def evaluate_coco(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    fig_test,  ax_test  = plt.subplots(figsize=(18,15))

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    #For CoCo
    results = []

    with torch.no_grad():
        # Batches
        for i, (image_vis, image_lwir, boxes, labels, index) in enumerate(tqdm(test_loader, desc='Evaluating')):

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs, predicted_scores = model(image_vis, image_lwir)
            
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            
            for box_t, label_t, score_t, ids in zip(det_boxes_batch ,det_labels_batch, det_scores_batch, index):
                for box, label, score in zip(box_t, label_t, score_t) :
                    bb = box.cpu().numpy().tolist()

                    # if score.item() > 0.1 :
                    results.append( {\
                                    'image_id': ids.item(), \
                                    'category_id': label.item(), \
                                    'bbox': [bb[0]*ori_size_out[1], bb[1]*ori_size_out[0], (bb[2]-bb[0])*ori_size_out[1], (bb[3]-bb[1])*ori_size_out[0]], \
                                    'score': score.item()} )
    
    rstFile = os.path.join(checkpoint_root, './COCO_TEST_det_{:s}.json'.format(checkpoint_name))            
    write_result_coco(results, rstFile)
    
    # rstFile = os.path.join('./jobs/2019-03-26_16h07m_[SSDPed_512x640][KAISTPed_train-all-02]video_make_test_full/SSDPed_512x640_epoch_0022_det.json')

    try:

        cocoDt = cocoGt.loadRes(rstFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds  = [1]    
        cocoEval.evaluate(0)
        cocoEval.accumulate()
        curPerf = cocoEval.summarize(0)    

        cocoEval.draw_figure(ax_test, rstFile.replace('json', 'jpg'))        
        #writer.add_scalars('LAMR/fppi', {'test': curPerf}, epoch)
        
        print('Recall: {:}'.format( 1-cocoEval.eval['yy'][0][-1] ) )

    except:
        import torchcv.utils.trace_error
        print('[Error] cannot evaluate by cocoEval. ')

if __name__ == '__main__':
    evaluate_coco(test_loader, model)
