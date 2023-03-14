import numpy as np
import torch
from torchvision import transforms as T
import tqdm
import pickle
import argparse

from keypoint.models import FRCNN, StackedHourglass, fasterrcnn_backbone
from keypoint.bop_dataset import BOPDataset
from keypoint.train.transforms import ToTensor, Normalize, AffineCrop

from utils import conformity_score, one_each

parser = argparse.ArgumentParser()
parser.add_argument('--score_type', action='store', type=str)
parser.add_argument('--do_frcnn', action='store_true')
args = parser.parse_args()

score_type  = args.score_type
do_frcnn    = args.do_frcnn

# Load dataset 
dataset_name = 'lmo' # this the lmo calibration dataset containing 200 images
root         = './keypoint/data/bop'
num_classes  = {'lmo':8, 'lmo-org':8} 
device       = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset      = BOPDataset(root, dataset_name, split='test', return_coco=True)
dataset._set_kpts_info()

if do_frcnn:
    # Load Faster-RCNN detector
    detector_trainsform = T.ToTensor()
    state_dict = torch.load('data/detect_checkpoints/d{}.pt'.format(dataset_name), map_location=device)['frcnn']
    detector = fasterrcnn_backbone('resnet101', num_classes=1+num_classes[dataset_name]).to(device)
    detector.eval()
    detector.load_state_dict(state_dict)

# Load keypoints detector
transform_list = []
transform_list.append(AffineCrop(out_size=256, scale_factor=0, rotation_factor=0, dialation=0.25))
transform_list.append(ToTensor())
transform_list.append(Normalize())
kpts_transform = T.Compose(transform_list)

state_dict    = torch.load('keypoint/data/kpts_checkpoints/{}.pt'.format(dataset_name), map_location=device)['stacked_hg']
kpts_detector = StackedHourglass(dataset.n_kpts).to(device)
kpts_detector.eval()
kpts_detector.load_state_dict(state_dict)

# useful info about dataset
n_kpts  = dataset.n_kpts
n_smps  = len(dataset)
obj2idx = dataset.obj2idx
idx2obj = {v:k for k,v in obj2idx.items()}
lab2obj = {v+1:k for k,v in obj2idx.items()}
n_objs  = len(idx2obj)

# Prepare to store obj scores
obj_scores = [[] for i in range(n_objs)]
print(f'nonconformity function: {score_type}.')

# Compute conformity score on calibration dataset
for i in tqdm.tqdm(range(n_smps)):
    sample      = dataset[i]
    meta        = dataset.db[i]

    image       = sample['image']
    gt_boxes    = sample['boxes']
    gt_objs     = [lab2obj[l] for l in sample['labels']]
    gt_kpts     = meta['keypoints']

    if do_frcnn:
        # Object detection
        with torch.no_grad():
            img = detector_trainsform(image).to(device)
            pred = detector([img])[0]
            pred = {k:v.cpu() for k,v in pred.items()}
        pd_boxes, pd_labels = one_each(pred, thresh=0.0)
        pd_objs = [lab2obj[i] for i in pd_labels.tolist()]
        pd_boxes = pd_boxes.tolist()
        
        _, comm1, comm2 = np.intersect1d(np.array(pd_objs), np.array(gt_objs), return_indices=True)
        comm1 = comm1.tolist()
        comm2 = comm2.tolist()

        pd_objs_true = [pd_objs[i] for i in comm1]
        pd_boxes_true = [pd_boxes[i] for i in comm1]
        gt_kpts_pd = [gt_kpts[i] for i in comm2]
        gt_objs = pd_objs_true
        gt_boxes = pd_boxes_true
        gt_kpts = gt_kpts_pd
        
    for obj, box, gt_kpt in zip(gt_objs, gt_boxes, gt_kpts):
        box         = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        gt_kpt_homo = np.concatenate(
            (gt_kpt,np.ones((gt_kpt.shape[0],1))),axis=1)
        input_crop  = {'image':image, 'bb':box, 'keypoints':gt_kpt_homo}
        input_crop  = kpts_transform(input_crop)
        gt_kpt_crop = input_crop['keypoints'][:,:2].numpy() * (64/256) # the heatmap is size (64,64), rescale from 256 to 64
        
        with torch.no_grad():
            batch = input_crop['image'][None].to(device)
            output = kpts_detector(batch)
            output = output[-1].cpu()
        
        kpt_start       = dataset.obj2kptid[obj][0]
        kpt_end         = dataset.obj2kptid[obj][1]
        heatmaps_pred   = torch.squeeze(
            output[[0], kpt_start:kpt_end, :, :])
        
        scores = []
        for j in np.arange(kpt_start,kpt_end):
            score = conformity_score(
                np.squeeze(gt_kpt_crop[j-kpt_start,:]),
                torch.squeeze(heatmaps_pred[j-kpt_start,:]).numpy(),
                type=score_type)
            scores.append(score)
        # @Apoorva: here is the place to quickly implement the windowed nonconformity score
        max_score = np.max(np.stack(scores))
        obj_scores[obj2idx[obj]].append(max_score)

obj_scores_np = []
for i in range(n_objs):
    obj_scores_np.append(np.array(obj_scores[i]))
fname = f'calibration_scores_{score_type}_{dataset_name}.pkl'
if do_frcnn:
    fname = f'calibration_scores_{score_type}_{dataset_name}_frcnn.pkl'
with open(fname, 'wb') as f:
    pickle.dump(obj_scores_np, f)