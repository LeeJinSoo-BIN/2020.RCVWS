import json
import os
import os.path
from PIL import Image, ImageDraw, ImageFont
from utils import *
import numpy as np

DB_ROOT = './datasets/New_Sejong_RCV_dataset'
image_set = 'train.txt'

imgpath_vis_New_Sejong = os.path.join('%s','RGBTDv3/Image','%s','RGB','I%s.png')
imgpath_lwir_New_Sejong = os.path.join('%s','RGBTDv3/Image','%s','Ther','I%s.png') 

### Load 

id_offset = 0
image_id_offset = 0

vis_total = 0
lwir_total = 0

vis_1_mean = 0
vis_2_mean = 0
vis_3_mean = 0
lwir_mean = 0

vis_1_std = 0
vis_2_std = 0
vis_3_std = 0
lwir_std = 0

ids = list() 

for line in open(os.path.join(DB_ROOT, 'RGBTDv3/ImageSet', image_set)):
    ids.append((DB_ROOT, line.strip().split('/')))
    
for ii, annotation_path in enumerate(ids):

    print(ii,len(ids))
    
    frame_id = ids[ii]

    vis = np.asarray(Image.open(imgpath_vis_New_Sejong%(DB_ROOT,frame_id[1][0],frame_id[1][1][-7:])))
    lwir = np.asarray(Image.open(imgpath_lwir_New_Sejong%(DB_ROOT,frame_id[1][0],frame_id[1][1][-7:])))

    vis = vis/255
    lwir = lwir/255

    vis_1_mean += vis[:,:,0].mean()
    vis_2_mean += vis[:,:,1].mean()
    vis_3_mean += vis[:,:,2].mean()
    lwir_mean += lwir.mean()

    vis_1_std += vis[:,:,0].std()
    vis_2_std += vis[:,:,1].std()
    vis_3_std += vis[:,:,2].std()
    lwir_std += lwir.std()

vis_1_mean = vis_1_mean/len(ids)
vis_2_mean = vis_2_mean/len(ids)
vis_3_mean = vis_3_mean/len(ids)
vis_1_std = vis_1_std/len(ids)
vis_2_std = vis_2_std/len(ids)
vis_3_std = vis_3_std/len(ids)

lwir_mean = lwir_mean/len(ids)
lwir_std = lwir_std/len(ids)

print(vis_1_mean, vis_2_mean, vis_3_mean)
print(vis_1_std, vis_2_std, vis_3_std)
print(lwir_mean, lwir_std)