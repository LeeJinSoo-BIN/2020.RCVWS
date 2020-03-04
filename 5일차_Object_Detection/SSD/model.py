from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
import torch as t
import pdb
import math

import torch.nn.init as init
from Correlation_Module.spatial_correlation_sampler import SpatialCorrelationSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):

    def __init__(self):
        super(VGGBase, self).__init__()

        self.conv1_1_vis = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_vis = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_vis = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_vis = nn.BatchNorm2d(64, affine=True)
        
        self.pool1_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_vis = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_vis = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_vis = nn.BatchNorm2d(128, affine=True)

        self.pool2_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_vis = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_vis = nn.BatchNorm2d(256, affine=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool4 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv6_1_bn = nn.BatchNorm2d(512, affine=True)  
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1)

        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        self.conv7_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv7_2_bn = nn.BatchNorm2d(512, affine=True)       

        self.load_pretrained_layers()

    def forward(self, image_vis):

        out_vis = F.relu(self.conv1_1_bn_vis(self.conv1_1_vis(image_vis)))  
        out_vis = F.relu(self.conv1_2_bn_vis(self.conv1_2_vis(out_vis)))
        out_vis = self.pool1_vis(out_vis)  

        out_vis = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis)))
        out_vis = F.relu(self.conv2_2_bn_vis(self.conv2_2_vis(out_vis))) 
        out_vis = self.pool2_vis(out_vis) 

        out_vis = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis))) 
        out_vis = F.relu(self.conv3_2_bn_vis(self.conv3_2_vis(out_vis))) 
        out_vis = F.relu(self.conv3_3_bn_vis(self.conv3_3_vis(out_vis)))

        out_vis = self.pool3(out_vis)

        out_vis = F.relu(self.conv4_1_bn(self.conv4_1(out_vis))) 
        out_vis = F.relu(self.conv4_2_bn(self.conv4_2(out_vis))) 
        out_vis = F.relu(self.conv4_3_bn(self.conv4_3(out_vis))) 
        out_vis = self.pool4(out_vis)

        conv4_3_feats = out_vis

        out_vis = F.relu(self.conv5_1_bn(self.conv5_1(out_vis))) 
        out_vis = F.relu(self.conv5_2_bn(self.conv5_2(out_vis))) 
        out_vis = F.relu(self.conv5_3_bn(self.conv5_3(out_vis))) 
        out_vis = self.pool5(out_vis)
        
        out_vis = F.relu(self.conv6_1_bn(self.conv6_1(out_vis))) 
        out_vis = F.relu(self.conv6_2(out_vis))
        conv6_feats = out_vis

        out_vis = F.relu(self.conv7_1(out_vis))
        out_vis = F.relu(self.conv7_2_bn(self.conv7_2(out_vis))) 
        conv7_feats = out_vis

        return conv4_3_feats, conv6_feats, conv7_feats

    def load_pretrained_layers(self):

        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(param_names[:49]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        for i, param in enumerate(param_names[49:-22]):    
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i+49]]

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")

    
    
class AuxiliaryConvolutions(nn.Module):

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True) 

        self.conv9_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv9_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)

        self.conv10_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv10_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)

        self.init_conv2d()


    def init_conv2d(self):

        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):

        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out)) 
        conv8_feats = out  

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_feats = out 

        out = F.relu(self.conv10_1(out)) 
        out = F.relu(self.conv10_2(out)) 
        conv10_feats = out  

        return conv8_feats, conv9_feats, conv10_feats


class PredictionConvolutions(nn.Module):


    def __init__(self, n_classes):

        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        n_boxes = {'conv4_3': 6,
                    'conv6': 6,
                    'conv7': 6,
                    'conv8': 6,
                    'conv9': 6,
                    'conv10': 6,}

        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv6 = nn.Conv2d(512, n_boxes['conv6'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(512, n_boxes['conv6'] * 4, kernel_size=3, padding=1)
        self.loc_conv8 = nn.Conv2d(512, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv9 = nn.Conv2d(512, n_boxes['conv8'] * 4, kernel_size=3, padding=1)
        self.loc_conv10 = nn.Conv2d(512, n_boxes['conv9'] * 4, kernel_size=3, padding=1)

        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv6 = nn.Conv2d(512, n_boxes['conv6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(512, n_boxes['conv6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8 = nn.Conv2d(512, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9 = nn.Conv2d(512, n_boxes['conv8'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10 = nn.Conv2d(512, n_boxes['conv9'] * n_classes, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats):

        batch_size = conv4_3_feats.size(0)

        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv6 = self.loc_conv6(conv6_feats) 
        l_conv6 = l_conv6.permute(0, 2, 3, 1).contiguous() 
        l_conv6 = l_conv6.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats) 
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous() 
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8 = self.loc_conv8(conv8_feats)
        l_conv8 = l_conv8.permute(0, 2, 3, 1).contiguous()  
        l_conv8 = l_conv8.view(batch_size, -1, 4)

        l_conv9 = self.loc_conv9(conv9_feats) 
        l_conv9 = l_conv9.permute(0, 2, 3, 1).contiguous()  
        l_conv9 = l_conv9.view(batch_size, -1, 4) 

        l_conv10 = self.loc_conv10(conv10_feats)
        l_conv10 = l_conv10.permute(0, 2, 3, 1).contiguous()
        l_conv10 = l_conv10.view(batch_size, -1, 4)  

        c_conv4_3 = self.cl_conv4_3(conv4_3_feats) 
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes) 


        c_conv6 = self.cl_conv6(conv6_feats)
        c_conv6 = c_conv6.permute(0, 2, 3, 1).contiguous() 
        c_conv6 = c_conv6.view(batch_size, -1, self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8 = self.cl_conv8(conv8_feats) 
        c_conv8 = c_conv8.permute(0, 2, 3, 1).contiguous()  
        c_conv8 = c_conv8.view(batch_size, -1, self.n_classes) 

        c_conv9 = self.cl_conv9(conv9_feats) 
        c_conv9 = c_conv9.permute(0, 2, 3, 1).contiguous() 
        c_conv9 = c_conv9.view(batch_size, -1, self.n_classes)

        c_conv10 = self.cl_conv10(conv10_feats) 
        c_conv10 = c_conv10.permute(0, 2, 3, 1).contiguous() 
        c_conv10 = c_conv10.view(batch_size, -1, self.n_classes) 

        locs = torch.cat([l_conv4_3, l_conv6, l_conv7, l_conv8, l_conv9, l_conv10], dim=1)  
        classes_scores = torch.cat([c_conv4_3, c_conv6, c_conv7, c_conv8, c_conv9, c_conv10],dim=1) 

        return locs, classes_scores

class Sejong_PD(nn.Module):

    def __init__(self, n_classes):
        super(Sejong_PD, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image_vis):

        conv4_3_feats, conv6_feats , conv7_feats= self.base(image_vis) 
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt() 
        conv4_3_feats = conv4_3_feats / norm  
        conv4_3_feats = conv4_3_feats * self.rescale_factors 

        conv8_feats, conv9_feats, conv10_feats = self.aux_convs(conv7_feats) 
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats)  # (N, 8732, 4), (N, 8732, n_classes)
        return locs, classes_scores

    def create_prior_boxes(self):

        fmap_dims = {'conv4_3': [80,64],
                     'conv6': [40,32],
                     'conv7': [20,16],
                     'conv8': [10,8],
                     'conv9': [10,8],
                     'conv10': [10,8]}

        scale_ratios = {'conv4_3': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv6': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv7': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv8': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv9': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv10': [1., pow(2,1/3.), pow(2,2/3.)]}


        aspect_ratios = {'conv4_3': [1/2., 1/1.],
                         'conv6': [1/2., 1/1.],
                         'conv7': [1/2., 1/1.],
                         'conv8': [1/2., 1/1.],
                         'conv9': [1/2., 1/1.],
                         'conv10': [1/2., 1/1.]}


        anchor_areas = {'conv4_3': [40*40.],
                         'conv6': [80*80.],
                         'conv7': [160*160.],
                         'conv8': [200*200.],
                         'conv9': [280*280.],
                         'conv10': [360*360.]} 

        fmaps = ['conv4_3', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10']

        prior_boxes = []



        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap][1]):
                for j in range(fmap_dims[fmap][0]):
                    cx = (j + 0.5) / fmap_dims[fmap][0]
                    cy = (i + 0.5) / fmap_dims[fmap][1]
                    for s in anchor_areas[fmap]:
                        for ar in aspect_ratios[fmap]: 
                            h = sqrt(s/ar)                
                            w = ar * h
                            for sr in scale_ratios[fmap]: 
                                anchor_h = h*sr/512.
                                anchor_w = w*sr/640.
                                prior_boxes.append([cx, cy, anchor_w, anchor_h])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device) 

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2) 

        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):

            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)) 

            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)

            for c in range(1, self.n_classes):

                class_scores = predicted_scores[i][:, c]  
                score_above_min_score = class_scores > min_score  
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  
                class_decoded_locs = decoded_locs[score_above_min_score]  

                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  
                class_decoded_locs = class_decoded_locs[sort_ind]


                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs) 
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)


                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue

                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes = torch.cat(image_boxes, dim=0) 
            image_labels = torch.cat(image_labels, dim=0) 
            image_scores = torch.cat(image_scores, dim=0)  
            n_objects = image_scores.size(0)

            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k] 
                image_labels = image_labels[sort_ind][:top_k] 

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  

class MultiBoxLoss(nn.Module):

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=4, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False, ignore_index=-1)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
    
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device) 
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  
        for i in range(batch_size):

            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],self.priors_xy)

            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
            _, prior_for_each_object = overlap.max(dim=1) 

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            overlap_for_each_prior[prior_for_each_object] = 1.

            label_for_each_prior = labels[i][object_for_each_prior] 
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  

            true_classes[i] = label_for_each_prior

            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        positive_priors = true_classes > 0 

        # LOCALIZATION LOSS
        if true_locs[positive_priors].shape[0] == 0:
            loc_loss = 0.
        else:
            loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        # CONFIDENCE LOSS
        n_positives = positive_priors.sum(dim=1) 

        n_hard_negatives = self.neg_pos_ratio * n_positives 

        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors) 

        conf_loss_pos = conf_loss_all[positive_priors]  
        
        conf_loss_neg = conf_loss_all.clone() 
        conf_loss_neg[positive_priors] = 0. 
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True) 
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  

        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives] 

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / ( 1e-10 + n_positives.sum().float() )

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss , conf_loss , loc_loss, n_positives

