import torch
import time
import sys
import math
from tqdm import tqdm
tqdm.monitor_interval = 0
from tensorboardX import SummaryWriter
from utils import CheckpointDataLoader, CheckpointSaver

class BaseTrainer:

    def __init__(self, options):
        self.options = options
        self.collate_fn = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self._init_fn() # define your model, optimizers etc.
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)
        
        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

        self.lr_scheduler = None
        self.exponential_scheduler = None
        if self.options.lr_decay < 1.0:
            self.exponential_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                            optimizer = self.optimizer,, 
                            gamma = self.options.lr_decay,
                            last_epoch = self.step_count-1)
            print('lr_decay/epoch:', self.options.lr_decay)

        if self.options.lr_schedule is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer = self.optimizer,
                            milestones = self.options.lr_schedule,
                            gamma = self.options.lr_gamma,
                            last_epoch = self.step_count-1)

            print('lr_schedule:', self.options.lr_schedule)

    def _init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    # @profile
    def train(self):

        self.endtime = time.time() + self.options.time_to_run
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train,
                                                     collate_fn=self.collate_fn)
            warmup_scheduler = None
            warmup_steps = self.options.warmup_steps
            if epoch == 0 and self.step_count == 0 and self.checkpoint is None:
                warmup_iters = warmup_steps
                warmup_factor = 1./warmup_steps
                warmup_scheduler = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=math.ceil(len(self.train_ds)/self.options.batch_size),
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                
                if time.time() < self.endtime:
                    out = self._train_step(batch)

                    self.step_count += 1

                    if self.step_count % self.options.summary_steps == 0:
                        self._train_summaries(batch, *out)

                    if self.step_count % self.options.test_steps == 0:
                        val_loss = self.test()

                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                        tqdm.write('Checkpoint saved')

                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

                
                if warmup_scheduler is not None:
                    warmup_scheduler.step()
                    if self.step_count > warmup_steps:
                        print('Setting warmup scheduler to none')
                        warmup_scheduler = None

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if self.exponential_scheduler is not None:
                self.exponential_scheduler.step()

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None

            # save checkpoint after each epoch
            self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count) 

        return

    def _get_lr(self):
        return next(iter(self.optimizers_dict.values())).param_groups[0]['lr']

    def _train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def _train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _save_summaries method')

    def test(self, input_batch):
        raise NotImplementedError('You need to provide a _test_step method')


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
