import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class RandomSampler(Sampler):

    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size']*checkpoint['batch_idx']:]
        else:
            self.dataset_perm = torch.randperm(len(self.data_source)).tolist()
            self.perm = self.dataset_perm

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)

class SequentialSampler(Sampler):

    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size']*checkpoint['batch_idx']:]
        else:
            self.dataset_perm = list(range(len(self.data_source)))
            self.perm = self.dataset_perm

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)

class CheckpointDataLoader(DataLoader):
    
    def __init__(self, dataset, checkpoint=None, batch_size=1,
                 shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None):

        if shuffle:
            sampler = RandomSampler(dataset, checkpoint)
        else:
            sampler = SequentialSampler(dataset, checkpoint)
        if checkpoint is not None:
            self.checkpoint_batch_idx = checkpoint['batch_idx']
        else:
            self.checkpoint_batch_idx = 0

        super(CheckpointDataLoader, self).__init__(dataset, sampler=sampler, shuffle=False, batch_size=batch_size,
                                                   pin_memory=pin_memory, timeout=timeout, worker_init_fn=None, 
                                                   collate_fn=collate_fn)
