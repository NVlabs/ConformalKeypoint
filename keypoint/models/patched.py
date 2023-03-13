import torch
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import fastrcnn_loss

#************************************************************************
# Patch RPN forward function to return loss during eval()
# when "targets" is provided
#************************************************************************
def rpn_forward(self,
            images,       # type: ImageList
            features,     # type: Dict[str, Tensor]
            targets=None  # type: Optional[List[Dict[str, Tensor]]]
            ):
    # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
    """
    Args:
        images (ImageList): images for which we want to compute the predictions
        features (OrderedDict[Tensor]): features computed from the images that are
            used for computing the predictions. Each tensor in the list
            correspond to different feature levels
        targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
            If provided, each element in the dict should contain a field `boxes`,
            with the locations of the ground-truth boxes.
    Returns:
        boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
            image.
        losses (Dict[Tensor]): the losses for the model during training. During
            testing, it is an empty dict.
    """
    # RPN uses all feature maps that are available
    features = list(features.values())
    objectness, pred_bbox_deltas = self.head(features)
    anchors = self.anchor_generator(images, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = \
        concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    losses = {}
    if self.training:
        assert targets is not None
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets)
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
    #************************************
    # Patch start
    #************************************
    elif targets is not None:
        assert targets is not None
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets)
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
    #************************************
    # Patch end
    #************************************

    return boxes, losses




#************************************************************************
# Patch ROIHeads forward function to return loss during eval()
# when "targets" is provided
# This function is reduced to only work for detection task (eg. frcnn)
#************************************************************************
def roi_forward(self,
            features,      # type: Dict[str, Tensor]
            proposals,     # type: List[Tensor]
            image_shapes,  # type: List[Tuple[int, int]]
            targets=None   # type: Optional[List[Dict[str, Tensor]]]
            ):
    # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
    """
    Args:
        features (List[Tensor])
        proposals (List[Tensor[N, 4]])
        image_shapes (List[Tuple[H, W]])
        targets (List[Dict])
    """
    if targets is not None:
        for t in targets:
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            floating_point_types = (torch.float, torch.double, torch.half)
            assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
            assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
            if self.has_keypoint():
                assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

    if self.training:
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    #************************************
    # Patch start
    #************************************
    elif targets is not None:
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    #************************************
    # Patch end
    #************************************
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    losses = {}
    if self.training:
        assert labels is not None and regression_targets is not None
        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets)
        losses = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg
        }
    else:
        #************************************
        # Patch start
        #************************************
        if labels is not None and regression_targets is not None:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        #************************************
        # Patch end
        #************************************

        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

    return result, losses

