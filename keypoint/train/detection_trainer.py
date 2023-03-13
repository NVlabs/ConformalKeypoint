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
from transforms import ColorJitter, ToTensor, \
                RandomHorizontalFlip, RandomGaussianBlur, RandomGrayscale

from models import FRCNN

class DetectionTrainer(BaseTrainer):

    def _init_fn(self):
        transform_list = []
        transform_list.append(ColorJitter(brightness=self.options.jitter, contrast=self.options.jitter, saturation=self.options.jitter, hue=self.options.jitter/4))
        transform_list.append(RandomGrayscale(0.2))
        transform_list.append(RandomGaussianBlur(kernel_size=7))
        transform_list.append(ToTensor())
        transform_list.append(RandomHorizontalFlip(0.5))

        test_transform_list = []
        test_transform_list.append(ToTensor())

        self.train_ds = BOPDataset(self.options.dataset_dir, self.options.dataset, split='train', 
                                    valid_objid = self.options.objid,
                                    return_keypoints=False, return_coco=True, 
                                    transform=transforms.Compose(transform_list))

        self.test_ds = BOPDataset(self.options.dataset_dir, self.options.dataset, split='test', 
                                    valid_objid = self.options.objid,
                                    return_keypoints=False, return_coco=True, 
                                    transform=transforms.Compose(test_transform_list))

        self.collate_fn = lambda batch: tuple(batch)
        

        num_classes = len(self.train_ds.obj2idx) + 1
        self.model = FRCNN(num_classes).to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.options.lr)

        # pack all models and optimizers in dictionaries to interact with the checkpoint saver
        self.models_dict = {'frcnn': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        # meter to track moving average
        self.loss_box_meter = AverageMeter()
        self.loss_class_meter = AverageMeter()

        print('Using device:', self.device)
        print('Using optimizer:', self.options.optimizer)
        print('Total number of classes:', num_classes)

    def _train_step(self, input_batch):
        # Force optimizer to use initial/reset learning rate, if specified
        if self.options.new_lr is True:
            for g in self.optimizer.param_groups:
                g['lr'] = self.options.lr
            self.options.new_lr = False

        # structure input_batch for torchvision detection module
        images = [s['image'].to(self.device) for s in input_batch]
        targets = [{k: v.to(self.device) for k, v in s.items() if k!='image'} for s in input_batch]

        # train step
        self.model.train()
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        # value
        loss_box = loss_dict['loss_box_reg'].cpu().item()
        loss_class = loss_dict['loss_classifier'].cpu().item()

        ## reset every 25k steps
        if self.step_count % 100 == 0:
            self.loss_box_meter.reset()
            self.loss_class_meter.reset()

        self.loss_box_meter.update(loss_box)
        self.loss_class_meter.update(loss_class)

        return self.loss_box_meter.avg, self.loss_class_meter.avg

    
    def _train_summaries(self, batch, loss_box, loss_class):
        self._get_summaries(batch, loss_box, loss_class, is_train=True)


    def test(self):
        test_data_loader = DataLoader(self.test_ds, batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.shuffle_test,
                                      collate_fn=self.collate_fn)

        self.model.always_return_loss = True
        
        test_loss_box = torch.tensor(0.0, device=self.device)
        test_loss_class = torch.tensor(0.0, device=self.device)
        for tstep, batch in enumerate(tqdm(test_data_loader, desc='Testing')):
            if time.time() < self.endtime:

                loss_box, loss_class = self._test_step(batch)

                test_loss_box += loss_box
                test_loss_class += loss_class
            else:
                tqdm.write('Testing interrupted at step ' + str(tstep))
                break

        test_loss_box /= (tstep+1)
        test_loss_class /= (tstep+1)

        self.model.always_return_loss = False
        self._get_summaries(batch, test_loss_box, test_loss_class, is_train=False)


        return

    def _test_step(self, input_batch):
        
        images = [s['image'].to(self.device) for s in input_batch]
        targets = [{k: v.to(self.device) for k, v in s.items() if k!='image'} for s in input_batch]

        self.model.eval()
        with torch.no_grad():
            loss_dict = self.model(images, targets)

        loss_box = loss_dict['loss_box_reg'].cpu().item()
        loss_class = loss_dict['loss_classifier'].cpu().item()

        return loss_box, loss_class


    def _get_summaries(self, batch, loss_box, loss_class, is_train):
        images = [s['image'].to(self.device) for s in batch]
        targets = [{k: v.to(self.device) for k, v in s.items() if k!='image'} for s in batch]

        image = images[0]
        target = targets[0]

        self.model.eval()
        self.model.always_return_loss = False
        with torch.no_grad():
            pred = self.model([image])[0]

        # ground truth
        gt_boxes = target['boxes'].int()
        gt_labels = target['labels'].int()
        gt_labels = [str(l) for l in gt_labels.tolist()]


        # prediction
        thresh = 0.80
        conf = pred['scores'] > thresh

        conf_scores = pred['scores'][conf]
        conf_boxes = pred['boxes'][conf].int()
        conf_labels = pred['labels'][conf].int()

        valid = torch.zeros_like(conf_labels).bool()
        unique_labels = torch.unique(conf_labels)
        for uni in unique_labels:
            p = (conf_labels==uni).nonzero(as_tuple=False).reshape(-1)
            valid[p[0]] = True

        pd_boxes = conf_boxes[valid]
        pd_labels = conf_labels[valid]
        pd_labels = [str(l) for l in pd_labels.tolist()]
        
        self._save_summaries(image, gt_boxes, gt_labels, pd_boxes, pd_labels, 
                            loss_box, loss_class, self.step_count, is_train=is_train) 


    def _save_summaries(self, image, gt_boxes, gt_labels, pd_boxes, pd_labels, 
                        loss_box, loss_class, step, is_train=True):

        prefix = 'train/' if is_train else 'test/'

        self.summary_writer.add_scalar(prefix + 'loss_box', loss_box, step)
        self.summary_writer.add_scalar(prefix + 'loss_class', loss_class, step)

        self.summary_writer.add_image_with_boxes(prefix + 'gt_boxes', image, gt_boxes, 
                                                 step, labels=gt_labels, dataformats='CHW')
        self.summary_writer.add_image_with_boxes(prefix + 'pd_boxes', image, pd_boxes, 
                                                 step, labels=pd_labels, dataformats='CHW')

        if is_train:
            self.summary_writer.add_scalar('lr', self._get_lr(), step)
        return



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
