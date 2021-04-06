#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .box_ops import box_xyxy_to_cxcywh


__all__ = ["E2EVID"]


@META_ARCH_REGISTRY.register()
class E2EVID(nn.Module):
    """Implement E2EVID"""
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.E2EVID.NUM_CLASSES
        self.num_proposals = cfg.MODEL.E2EVID.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.E2EVID.HIDDEN_DIM
        self.num_heads = cfg.MODEL.E2EVID.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.E2EVID.CLASS_WEIGHT
        giou_weight = cfg.MODEL.E2EVID.GIOU_WEIGHT
        l1_weight = cfg.MODEL.E2EVID.L1_WEIGHT
        no_object_weight = cfg.MODEL.E2EVID.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.E2EVID.DEEP_SUPERVISION

        # Build Criterion.
        matcher = HungarianMatcher(
            cfg=cfg,
            cost_class=class_weight, 
            cost_bbox=l1_weight, 
            cost_giou=giou_weight,
        )
        weight_dict = {
            "loss_ce": class_weight, 
            "loss_bbox": l1_weight, 
            "loss_giou": giou_weight
        }
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(
            cfg=cfg,
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std  # TODO imagenet normalizer
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one video.
                For now, each item in the list is a list of dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        assert len(batched_inputs) == 1
        dataset_dict = batched_inputs[0]
        images, images_xywh = self.preprocess_image(dataset_dict)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = images_xywh[:, None, :].repeat(1, self.num_proposals, 1)

        # Prediction.
        outputs_class, outputs_coord = self.head(
            features, proposal_boxes, self.init_proposal_features.weight
        )
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.deep_supervision:
                output['aux_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        scores = torch.sigmoid(box_cls)
        labels = torch.arange(self.num_classes, device=self.device).\
                    unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

        for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, box_pred, image_sizes
        )):
            result = Instances(image_size)
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(100, sorted=False)
            labels_per_image = labels[topk_indices]
            box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1)
            box_pred_per_image = box_pred_per_image.view(-1, 4)[topk_indices]

            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results

    def preprocess_image(self, dataset_dict):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in dataset_dict]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_xywh = list()
        for bi in dataset_dict:
            h, w = bi["image"].shape[-2:]
            images_xywh.append(torch.tensor([0, 0, w, h], dtype=torch.float32, device=self.device))
        images_xywh = torch.stack(images_xywh)

        return images, images_xywh


"""
"v_QOlSCBRmfWY": {
  "segments": {
    "0": {
      "objects": [
        {"xbr": 719, "ybr": 404, "noun_phrases": [["a room", [7, 8]]], "ytl": 1, "frame_ind": 6, "xtl": 0, "crowds": 0}, 
        {"xbr": 412, "ybr": 314, "noun_phrases": [["a young woman", [0, 1, 2]], ["her", [12]]], "ytl": 76, "frame_ind": 6, "xtl": 290, "crowds": 0}
      ]
    }, 
    "1": {
      "objects": [
        {"xbr": 718, "ybr": 404, "noun_phrases": [["the room", [4, 5]]], "ytl": 0, "frame_ind": 6, "xtl": 0, "crowds": 0}, 
        {"xbr": 476, "ybr": 386, "noun_phrases": [["the girl", [0, 1]], ["her", [10]]], "ytl": 132, "frame_ind": 6, "xtl": 221, "crowds": 0}
      ]
    }, 
    "2": {
      "objects": [
        {"xbr": 718, "ybr": 403, "noun_phrases": [["the floor", [11, 12]]], "ytl": 256, "frame_ind": 9, "xtl": 2, "crowds": 0}, 
        {"xbr": 541, "ybr": 323, "noun_phrases": [["she", [0]]], "ytl": 279, "frame_ind": 9, "xtl": 352, "crowds": 0}, 
        {"xbr": 717, "ybr": 403, "noun_phrases": [["the room", [4, 5]]], "ytl": 0, "frame_ind": 9, "xtl": 0, "crowds": 0}
      ]
    }
  }, 
  "rwidth": 720, 
  "rheight": 405
}
"""
