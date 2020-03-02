import json
import os
import os.path
filename = 'Potenit_ALL.json'
cat_id_path = 'catids.json'
DB_ROOT = './datasets/New_Sejong_RCV_dataset/RGBTDv3'
image_set = 'test_all.txt'

annopath_Sejong_New = os.path.join('%s','json','%s','RGB','%s.json')
data = dict(annotations=[], images=[], categories=[])
### Load 
id_offset = 0
image_id_offset = 0
ids = list() 

for line in open(os.path.join(DB_ROOT, 'ImageSet', image_set)):
    ids.append((DB_ROOT, line.strip().split('/')))
    
for ii, annotation_path in enumerate(ids):
    
    frame_id = ids[ii]
    with open(annopath_Sejong_New %(DB_ROOT,frame_id[1][0],frame_id[1][1])) as j:
        data_t = json.load(j)
    for ann in data_t['annotation']:
        ann['category_id'] = int(ann['category_id'])
        ann['id'] += id_offset
        ann['image_id'] = image_id_offset
        ann['bbox'][2] = ann['bbox'][2]-ann['bbox'][0]
        ann['bbox'][3] = ann['bbox'][3]-ann['bbox'][1]
        if ann['occlusion'] == 2 or ann['bbox'][2] < 0 or ann['bbox'][3] < 0 :
            ann['ignore'] = 1
    
    id_offset += len(data_t['annotation'])
    data['annotations'].extend(data_t['annotation'])
    data_t['image'] = data_t['image'][0]
    data_t['image']['id'] = int(image_id_offset)
    data['images'].append([data_t['image']])
    
    image_id_offset += 1
with open(os.path.join('./datasets/New_Sejong_RCV_dataset_jw',cat_id_path), 'r') as f:
    data_t = json.load(f)  
data['categories'].extend(data_t)
print('Write results in COCO format.')
with open(filename, 'wt') as f:
    f.write( json.dumps(data, indent=4) )