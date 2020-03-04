#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
from scipy.io import loadmat
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def read_vbb(path,save_path=None, image_size=None):

    """
    Read the data of a .vbb file to a dictionary.

    pos  - [l t w h]: bb indicating predicted object extent
    l - pixel number from left of image
    t - pixel number from top of image
    w - width of bounding box
    h - height of bounding box

    """
    assert path[-3:] == 'vbb'

    # vbb 파일을 load
    vbb = loadmat(path)
    # 작업한 seq 파일의 총 프레임
    nFrame = int(vbb['A'][0][0][0][0][0])
    # bounding box list
    objLists = vbb['A'][0][0][1][0]
    # bounding box label list
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    
    data_box = dict(annotation=[], image=[])

    offset = 0
    
    # 프레임마다 box정보를 json 형식으로 저장
    # Label은 person 이라면 1, 아니라면 -1
    # 프레임에 bounding box가 없다면 기본값으로 저장
    for frame_id, obj in enumerate(objLists):

        frame_id += offset

        if obj.shape[1] > 0:

            for id, (bbox, occl) in enumerate(zip(obj['pos'][0],obj['occl'][0])):
                
                p = bbox[0].tolist()
                check = int(p[0])+int(p[1])+int(p[2])+int(p[3])
                bbox = [int(p[0] - 1), int(p[1] - 1), int(p[0]+p[2]), int(p[1]+p[3])]  # MATLAB is 1-origin

                occl = int(occl[0][0])
                
                datum = dict(bbox=[],category_id=[],id=[],image_id=[],iscrowd=[])
                datum_image = dict(file_name=[],height=[],id=[],width=[])
                
                if occl == 4 : 
                    continue
                if check == 308 :
                    import pdb;pdb.set_trace()

                datum['bbox'] = bbox

                if str(objLbl[id]) is "person" or "object" :
                    datum['category_id'] = 1
                else :
                    datum['category_id'] = -1

                datum['id'] = id
                datum['image_id'] = frame_id
                datum["occlusion"] = occl
                datum['iscrowd'] = 0
                data_box['annotation'].append(datum)
        
        else :
            id = 0
            p = [0,0,0,0]
            bbox = [0,0,0,0]
            occl = 0
            label = -1

            datum = dict(bbox=[],category_id=[],id=[],image_id=[],iscrowd=[])
            datum_image = dict(file_name=[],height=[],id=[],width=[])

            datum['bbox'] = bbox
            datum['category_id'] = -1
            datum['id'] = id
            datum['image_id'] = frame_id
            datum['iscrowd'] = 0
            datum['occlusion'] = occl

            data_box['annotation'].append(datum)
        
        datum_image['file_name'] = "RGB_L_{}".format(str(frame_id).zfill(7))
        datum_image['height'] = image_size[0]
        datum_image['id'] = frame_id
        datum_image['width'] = image_size[1]
        data_box['image'].append(datum_image)

        anno_fname = "RGB_L_{}.json".format(str(frame_id).zfill(7))
        anno_path = os.path.join(save_path, anno_fname)
        print("RGB_L_{}.json".format(str(frame_id).zfill(7)))

        try:
            with open(anno_path, 'w') as file_cache:
                json.dump(  data_box,
                            file_cache,
                            sort_keys=False,
                            indent=4,
                            ensure_ascii=False)
        except IOError:
            raise IOError('Unable to open file: {}'.format(anno_path))

        data_box = dict(annotation=[], image=[])


if __name__ == '__main__':

    # default parameter
    # RGB_L_{}.json 형식으로 저장됩니다
    image_size=[422,538] # [height,width]
    save_path ="Set03_json"
    read_vbb("Set03.vbb",save_path=save_path,image_size=image_size)