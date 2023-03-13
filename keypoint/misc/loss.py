import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class KptsMSELoss(nn.Module):
    def __init__(self, use_vis=False):
        super(KptsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_vis = use_vis

    def forward(self, output, target, vis):
        '''
        output: (BN, K, w, h)
        target: (BN, K, w, h)
        vis: (BN, K)
        '''
        batch_size = output.size(0)
        num_kpts = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_kpts, -1))
        heatmaps_gt = target.reshape((batch_size, num_kpts, -1))
        vis = vis.reshape((batch_size, num_kpts, 1))

        if self.use_vis:
            loss = self.criterion(
                heatmaps_pred.mul(vis),
                heatmaps_gt.mul(vis)
                )
        else:
            loss = self.criterion(heatmaps_pred, heatmaps_gt)

        return loss 





