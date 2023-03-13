import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

import time
from tqdm import tqdm
tqdm.monitor_interval = 0

from bop_dataset import BOPDataset
from base_trainer import BaseTrainer
from transforms import RandomFlipLR, RandomRescaleBB, RandomGrayscale, RandomRotation,\
                                RandomBlur, ColorJitter, CropAndResize, LocsToHeatmaps,\
                                ToTensor, Normalize, Denormalize, Select, AffineCrop
from models import StackedHourglass
from misc import Pose2DEval, KptsMSELoss

class KeypointTrainer(BaseTrainer):

    def _init_fn(self):
        transform_list = []
        transform_list.append(ColorJitter(brightness=self.options.jitter, contrast=self.options.jitter, saturation=self.options.jitter, hue=self.options.jitter/4))
        transform_list.append(AffineCrop(out_size=self.options.crop_size, scale_factor=0.15, rotation_factor=45, dialation=0.25))
        transform_list.append(LocsToHeatmaps(out_size=(self.options.heatmap_size, self.options.heatmap_size)))
        transform_list.append(ToTensor())
        transform_list.append(Normalize())

        test_transform_list = []
        test_transform_list.append(AffineCrop(out_size=self.options.crop_size, scale_factor=0, rotation_factor=0, dialation=0.25))
        test_transform_list.append(LocsToHeatmaps(out_size=(self.options.heatmap_size, self.options.heatmap_size)))
        test_transform_list.append(ToTensor())
        test_transform_list.append(Normalize())

        self.train_ds = BOPDataset(self.options.dataset_dir, self.options.dataset, split='train', 
                                    valid_objid = self.options.objid,
                                    return_keypoints=True, 
                                    transform=transforms.Compose(transform_list))

        self.test_ds = BOPDataset(self.options.dataset_dir, self.options.dataset, split='test', 
                                    valid_objid = self.options.objid,
                                    return_keypoints=True, 
                                    transform=transforms.Compose(test_transform_list))
        self.collate_fn = None

        self.options.num_keypoints = self.train_ds.n_kpts

        self.model = StackedHourglass(self.options.num_keypoints).to(self.device)

        if self.options.optimizer is 'adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.options.lr, 
                                        betas=(0.9, 0.999), eps=1e-08)
            print('Using ADAM.')
        else:
            self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=self.options.lr, 
                                        momentum=0, weight_decay=self.options.wd)
        

        # pack all models and optimizers in dictionaries to interact with the checkpoint saver
        self.models_dict = {'stacked_hg': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        self.criterion = KptsMSELoss(use_vis=self.options.use_vis).to(self.device)
        self.pose = Pose2DEval(detection_thresh=self.options.detection_thresh, dist_thresh=self.options.dist_thresh)

        print('Total number of model parameters:', self.model.num_trainable_parameters())
        print('Using device:', self.device)
        print('Using optimizer:', self.options.optimizer)

    def _train_step(self, input_batch):
        # Force optimizer to use initial/reset learning rate, if specified
        if self.options.new_lr is True:
            for g in self.optimizer.param_groups:
                g['lr'] = self.options.lr
            self.options.new_lr = False

        input_batch = {k: v.to(self.device) for k,v in input_batch.items()}

        self.model.train()
        images = input_batch['image']
        gt_keypoints = input_batch['keypoint_heatmaps']
        vis = input_batch['visible_keypoints']

        pred_keypoints = self.model(images)
        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(pred_keypoints)):
            loss += self.criterion(pred_keypoints[i], gt_keypoints, vis)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return [pk.detach() for pk in pred_keypoints], loss.detach()
    
    def _train_summaries(self, batch, pred_keypoints, loss):
        batch = {k: v.to(self.device) for k,v in batch.items()}
        
        pck = self.pose.pck(batch['keypoint_heatmaps'], pred_keypoints[-1])
        self._save_summaries(batch, pred_keypoints, loss, pck, self.step_count, is_train=True) 

    def test(self):
        test_data_loader = DataLoader(self.test_ds, batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.shuffle_test)
        test_loss = torch.tensor(0.0, device=self.device)
        mean_pck = 0.0
        for tstep, batch in enumerate(tqdm(test_data_loader, desc='Testing')):
            if time.time() < self.endtime:
                batch = {k: v.to(self.device) for k,v in batch.items()}
                pred_keypoints, loss = self._test_step(batch)
                test_loss += loss.data
                mean_pck += self.pose.pck(batch['keypoint_heatmaps'], pred_keypoints[-1])
            else:
                tqdm.write('Testing interrupted at step ' + str(tstep))
                break
        test_loss /= (tstep+1)
        mean_pck /= (tstep+1)
        self._save_summaries(batch, pred_keypoints, test_loss, mean_pck, self.step_count, is_train=False) 
        return test_loss

    def _test_step(self, input_batch):
        self.model.eval()
        images = input_batch['image']
        gt_keypoints = input_batch['keypoint_heatmaps']
        vis = input_batch['visible_keypoints']
        with torch.no_grad():
            pred_keypoints = self.model(images)
        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(pred_keypoints)):
            loss += self.criterion(pred_keypoints[i], gt_keypoints, vis)
        return pred_keypoints, loss

    def _save_summaries(self, input_batch, pred_keypoints, loss, pck, step, is_train=True):
        prefix = 'train/' if is_train else 'test/'
        input_batch = Denormalize()(input_batch)
        images = input_batch['image']
        gt_keypoints = input_batch['keypoint_heatmaps']

        gt_image_keypoints = []
        pred_image_keypoints = []
        gt_image_keypoints, pred_image_keypoints = self.pose.draw_keypoints_with_labels(images, gt_keypoints, pred_keypoints[-1])

        gt_image_keypoints_grid = make_grid(gt_image_keypoints, pad_value=1, nrow=3)
        pred_image_keypoints_grid = make_grid(pred_image_keypoints, pad_value=1, nrow=3)

        pred_heatmaps_grid = make_grid(pred_keypoints[-1][0,:,:,:].unsqueeze(0).transpose(0,1), pad_value=1, nrow=5)
        pred_heatmaps_grid[pred_heatmaps_grid > 1] = 1
        pred_heatmaps_grid[pred_heatmaps_grid < 0] = 0

        self.summary_writer.add_scalar(prefix + 'loss', loss, step)
        self.summary_writer.add_scalar(prefix + 'PCK', pck, step)
        self.summary_writer.add_image(prefix + 'gt_image_keypoints', gt_image_keypoints_grid, step)
        self.summary_writer.add_image(prefix + 'pred_image_keypoints', pred_image_keypoints_grid, step)
        self.summary_writer.add_image(prefix + 'pred_heatmaps_image1', pred_heatmaps_grid, step)
        if is_train:
            self.summary_writer.add_scalar('lr', self._get_lr(), step)
        return
