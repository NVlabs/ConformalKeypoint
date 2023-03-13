#!/usr/bin/python3

# this is a hack to make it work in the cluster because
#import matplotlib
#matplotlib.use('Agg')

import torch
import numpy as np
from train_options import TrainOptions
from keypoint_trainer import KeypointTrainer
from detection_trainer import DetectionTrainer

if __name__ == '__main__':

	# reproducibility
	np.random.seed(0)
	torch.manual_seed(0)

	# training code
    options = TrainOptions().parse_args()
    if options.task == 'keypoints':
        trainer = KeypointTrainer(options)
    elif options.task == 'detection':
        trainer = DetectionTrainer(options)
    else:
        print("The requested option is not supported on this dataset")
        exit()

    trainer.train()
