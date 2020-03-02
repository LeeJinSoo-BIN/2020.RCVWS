from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import os
import os.path
import pdb
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './jobs/2020-02-11_17h20m_train_SF+_POTENIT_Hard_OCc/checkpoint_ssd300.pth.tar026'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.' % (start_epoch))
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((512, 640))
to_tensor = transforms.ToTensor()
normalize_vis = transforms.Normalize(mean=[0.5873, 0.5328, 0.4877],std=[0.2331, 0.2160, 0.2010])
normalize_lwir = transforms.Normalize(mean=[0.4126],std=[0.1453])
ori_size_out = [442,538]

#Data load
DB_ROOT = './datasets/New_Sejong_RCV_dataset'
image_set = 'test_all.txt'

imgpath_vis_New_Sejong = os.path.join('%s','RGBTDv3/Image','%s','RGB','I%s.png')
imgpath_lwir_New_Sejong = os.path.join('%s','RGBTDv3/Image','%s','Ther','I%s.png') 

ids = list() 

for line in open(os.path.join(DB_ROOT, 'RGBTDv3/ImageSet', image_set)):
    ids.append((DB_ROOT, line.strip().split('/')))

def detect(original_image, lwir_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image_vis = normalize_vis(to_tensor(original_image))
    image_lwir = normalize_lwir(to_tensor(lwir_image))

    # Move to default device
    image_vis = image_vis.to(device)
    image_lwir = image_lwir.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image_vis.unsqueeze(0),image_lwir.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)  
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)


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
        text_score = str(det_scores[0][i])[:7]
        text_size = font.getsize(text_score)
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        # draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
        draw.text(xy=text_location, text='{:.4f}'.format(det_scores[0][i].item()), fill='white', font=font)
    # del draw
    
    return annotated_image


if __name__ == '__main__':


    for ii, annotation_path in enumerate(ids):
        
        frame_id = ids[ii]

        # vis = resize(Image.open(imgpath_vis_New_Sejong%(DB_ROOT,frame_id[1][0],frame_id[1][1][-7:])))
        # lwir = resize(Image.open(imgpath_lwir_New_Sejong%(DB_ROOT,frame_id[1][0],frame_id[1][1][-7:])))
        vis = resize(Image.open('./datasets/kaist-rgbt/images/set08/V002/visible/I00491.jpg'))
        lwir = resize(Image.open('./datasets/kaist-rgbt/images/set08/V002/lwir/I00491.jpg')).convert('L')

        annotate_lwir = detect(vis, lwir, min_score=0.8, max_overlap=0.5, top_k=200)
        annotate_lwir.save('./Detection_visualization/I{:06d}.jpg'.format(ii))
        print('%d.jpg saved' %ii)
  