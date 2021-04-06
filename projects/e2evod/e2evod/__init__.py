#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_e2evid_config
from .dataset import register_vid_instances
from .dataset_mapper import VIDDatasetMapper
from .detector import E2EVID
from .evaluator import VIDEvaluator
