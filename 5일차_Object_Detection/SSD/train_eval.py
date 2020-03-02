import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data as data
import argparse

from model import SSD300, MultiBoxLoss
from datasets import *
from utils import *
from torchcv.datasets.transforms import *
import torch.nn.functional as F
from torchcv.utils import run_tensorboard
from tensorboardX import SummaryWriter
import numpy as np
import logging
import logging.handlers
from datetime import datetime
from tqdm import tqdm
from pprint import PrettyPrinter
import torch
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
from torchcv.evaluations.coco import COCO
from torchcv.evaluations.eval_MR_multisetup import COCOeval

#############################################################################3

annType = 'bbox'
DB_ROOT = './datasets/New_Sejong_RCV_dataset/RGBTDv3'
JSON_GT_FILE = os.path.join( DB_ROOT, 'Potenit_20.json' )
cocoGt = COCO(JSON_GT_FILE)
input_size = [512, 640]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = None 
batch_size = 4 
start_epoch = 0  
epochs = 40 
epochs_since_improvement = 0  
best_loss = 100.  
workers = 10
print_freq = 10 
lr = 1e-3  
momentum = 0.9 
weight_decay = 5e-4 
grad_clip = None 
port = 8807
cudnn.benchmark = True
ori_size = (422, 538)
ori_size_out = [422,538]

# random seed fix 
torch.manual_seed(9)
torch.cuda.manual_seed(9)
np.random.seed(9)
random.seed(9)
torch.backends.cudnn.deterministic=True

parser = argparse.ArgumentParser(description='RCV_Winter_School')
parser.add_argument('--exp_time',   default=None, type=str,  help='set if you want to use exp time')
parser.add_argument('--exp_name',   default=None, type=str,  help='set if you want to use exp name')
parser.add_argument('--synth_fail',         default=['None', 'None'], nargs='+', type=str, help='Specify synthetic failure: e.g. crack1.jpg None')

args = parser.parse_args()

def main():
    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint
    if checkpoint is None:
        model = Sejong_PD(n_classes=n_classes)
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ int(epochs*0.5), int(epochs*0.75) ], gamma=0.1)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        train_loss = checkpoint['loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ int(epochs*0.5), int(epochs*0.75) ], gamma=0.1)

    if not args.synth_fail == ['None', 'None']:
        fail_mask = [ os.path.join('synthetic_failure_masks', mask ) for mask in args.synth_fail ]
        preprocess1.add( [ SynthFail(fail_mask, (ori_size)) ] )

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    preprocess2 = Compose([  ColorJitter(0.3, 0.3, 0.3), ColorJitterLWIR(contrast=0.3),  FaultTolerant([0.5, 0.5])])

    transforms2 = Compose([ RandomHorizontalFlip(), \
                            RandomResizedCrop( [512,640], scale=(0.25, 4.0), ratio=(0.8, 1.2)), \
                            ToTensor(), \
                            Normalize( [0.5873,0.5328,0.4877], [0.2331,0.2160,0.2010], 'R'), \
                            Normalize( [0.4126], [0.1453], 'T')])

    preprocess1 = Compose([ ])    
    transforms1 = Compose([ Resize(input_size), \
                            ToTensor(), \
                            Normalize( [0.5873,0.5328,0.4877], [0.2331,0.2160,0.2010], 'R'), \
                            Normalize( [0.4126], [0.1453], 'T')])

    train_dataset = Sejong_Ped('train.txt',img_transform=preprocess2, co_transform=transforms2, condition='train')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers,
                                               collate_fn=train_dataset.collate_fn, 
                                               pin_memory=True)  # note that we're passing the collate function here

    test_dataset = Sejong_Ped('test_20.txt',img_transform=preprocess1, co_transform=transforms1, condition='test')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=workers,
                                                collate_fn=test_dataset.collate_fn, 
                                                pin_memory=True)    

    #############################################################################################################################
    
    ### Set job directory

    if args.exp_time is None:
        args.exp_time        = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    
    exp_name        = ('_' + args.exp_name) if args.exp_name else '_' 
    jobs_dir        = os.path.join( 'jobs', args.exp_time + exp_name )
    args.jobs_dir   = jobs_dir

    snapshot_dir    = os.path.join( jobs_dir, 'snapshots' )
    tensorboard_dir    = os.path.join( jobs_dir, 'tensorboardX' )
    if not os.path.exists(snapshot_dir):        os.makedirs(snapshot_dir)
    if not os.path.exists(tensorboard_dir):     os.makedirs(tensorboard_dir)
    run_tensorboard( tensorboard_dir, port )
    
    import tarfile
    tar = tarfile.open( os.path.join(jobs_dir, 'sources.tar'), 'w' )
    tar.add( 'torchcv' )    
    tar.add( __file__ )

    import glob
    for file in sorted( glob.glob('*.py') ):
        tar.add( file )

    tar.close()

    writer = SummaryWriter(os.path.join(jobs_dir, 'tensorboardX'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(levelname)s] [%(asctime)-11s] %(message)s')
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    logger.addHandler(h)

    h = logging.FileHandler(os.path.join(jobs_dir, 'log_{:s}.txt'.format(args.exp_time)))
    h.setFormatter(fmt)
    logger.addHandler(h)

    settings = vars(args)
    for key, value in settings.items():
        settings[key] = value   

    logger.info('Exp time: {}'.format(settings['exp_time']))
    for key, value in settings.items():
        if key == 'exp_time':
            continue
        logger.info('\t{}: {}'.format(key, value))

    logger.info('Preprocess for training')
    logger.info( preprocess2 )
    logger.info('Transforms for training')
    logger.info( transforms2 )

    #############################################################################################################################

    for epoch in range(start_epoch, epochs):

        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger,
                           writer=writer)

        optim_scheduler.step()

        writer.add_scalars('train/epoch', {'epoch_train_loss': train_loss},global_step=epoch )
        save_checkpoint(epoch, model, optimizer, train_loss, jobs_dir)
        
        if epoch >= 1 :
            evaluate_coco(test_loader, model, epoch,jobs_dir,writer)
            



def train(train_loader, model, criterion, optimizer, epoch, logger, writer):

    model.train() 

    batch_time = AverageMeter()
    data_time = AverageMeter() 
    losses = AverageMeter()  
    losses_loc = AverageMeter()  
    losses_cls = AverageMeter() 
    start = time.time()

    for batch_idx, (image_vis, image_lwir, boxes, labels, _) in enumerate(train_loader):

        data_time.update(time.time() - start)

        image_vis = image_vis.to(device) 

        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(image_vis)  
        
        loss,cls_loss,loc_loss,n_positives = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        optimizer.zero_grad()
        loss.backward()

        if np.isnan(loss.item()):
            import pdb; pdb.set_trace()
            loss,cls_loss,loc_loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        losses.update(loss.item())
        losses_loc.update(loc_loss)
        losses_cls.update(cls_loss)
        batch_time.update(time.time() - start)

        start = time.time()

        if batch_idx and batch_idx % print_freq == 0:
            import pdb;         
            writer.add_scalars('train/loss', {'loss': losses.avg}, global_step=epoch*len(train_loader)+batch_idx )
            writer.add_scalars('train/loc', {'loss': losses_loc.avg}, global_step=epoch*len(train_loader)+batch_idx )                
            writer.add_scalars('train/cls', {'loss': losses_cls.avg}, global_step=epoch*len(train_loader)+batch_idx )

        if batch_idx % print_freq == 0:

            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'num of Positive {Positive}\t'.format(epoch, batch_idx, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, Positive=n_positives))

    del predicted_locs, predicted_scores, image_vis, image_lwir, boxes, labels
    return  losses.avg

def evaluate_coco(test_loader, model,epoch,jobs_dir,writer):

    fig_test,  ax_test  = plt.subplots(figsize=(18,15))

    model.eval()

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    results = []

    with torch.no_grad():

        for i, (image_vis, image_lwir, boxes, labels, index) in enumerate(tqdm(test_loader, desc='Evaluating')):

            image_vis = image_vis.to(device)

            predicted_locs, predicted_scores = model(image_vis)
            
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.45,
                                                                                       top_k=50)

            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            
            for box_t, label_t, score_t, ids in zip(det_boxes_batch ,det_labels_batch, det_scores_batch, index):
                for box, label, score in zip(box_t, label_t, score_t) :
                    bb = box.cpu().numpy().tolist()

                    results.append( {\
                                    'image_id': ids.item(), \
                                    'category_id': label.item(), \
                                    'bbox': [bb[0]*ori_size_out[1], bb[1]*ori_size_out[0], (bb[2]-bb[0])*ori_size_out[1], (bb[3]-bb[1])*ori_size_out[0]], \
                                    'score': score.item()} )

    rstFile = os.path.join(jobs_dir, './COCO_TEST_det_{:d}.json'.format(epoch))            
    write_result_coco(results, rstFile)

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
        writer.add_scalars('LAMR/fppi', {'test': curPerf}, epoch)
        
        print('Recall: {:}'.format( 1-cocoEval.eval['yy'][0][-1] ) )

    except:
        import torchcv.utils.trace_error
        print('[Error] cannot evaluate by cocoEval. ')

if __name__ == '__main__':

    main()
