import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .patched import rpn_forward, roi_forward
import types

@torch.jit.unused
def eager_outputs(self, losses, detections):
	if self.training or self.always_return_loss:
		return losses

	return detections


def FRCNN(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # override ouput functions
    model.always_return_loss = False

    model.rpn.forward = types.MethodType(rpn_forward, model.rpn)
    model.roi_heads.forward = types.MethodType(roi_forward, model.roi_heads)
    model.eager_outputs = types.MethodType(eager_outputs, model)

    return model


def fasterrcnn_backbone(backbone_name='resnet50',
            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    '''
    Input:
    backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
    'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
    '''

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # switch backbone
    backbone = resnet_fpn_backbone(backbone_name, pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model.backbone = backbone


    # override ouput functions
    model.always_return_loss = False
    model.rpn.forward = types.MethodType(rpn_forward, model.rpn)
    model.roi_heads.forward = types.MethodType(roi_forward, model.roi_heads)
    model.eager_outputs = types.MethodType(eager_outputs, model)

    return model


