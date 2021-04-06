# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen

__all__ = ["VIDDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class VIDDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by SparseRCNN.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors

    """
    def __init__(self, cfg, is_train=True):
        self.crop_gen = [
            T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
            T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
        ]

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), 
            str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of ONE video, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept

        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        if self.is_train:
            dataset_dict = # TODO: sample a fixed number of frames

        new_dataset_dict = []
        for item in dataset_dict:
            image = utils.read_image(item["filename"], format=self.img_format)
            utils.check_image_size(item, image)

            # TODO: SSD random crop
            image, transforms = T.apply_transform_gens(
                self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
            )

            image_shape = image.shape[:2]  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            sample = {"image": image}

            if not self.is_train:
                new_dataset_dict.append(sample)
                continue

            # USER: Implement additional transformations if you have other types of data
            boxes = [
                utils.transform_instance_annotations(box, transforms, image_shape)
                for box in item["boxes"]
            ]
            instances = # 
            sample["instances"] = instances
            new_dataset_dict.append(sample)

        return new_dataset_dict
