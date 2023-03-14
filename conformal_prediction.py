import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
import tqdm
import pickle, argparse
import os

from keypoint.models import FRCNN, StackedHourglass, fasterrcnn_backbone
from keypoint.bop_dataset import BOPDataset
from keypoint.train.transforms import ToTensor, Normalize, AffineCrop
from keypoint.misc.pose2d_eval import Pose2DEval

from utils import icp, draw_icp_ball, draw_icp_ellipse

def heatmap2org(kpts,lams,T):
    '''
    The heatmap is on the cropped image, this function converts the prediction sets on the cropped image to the original image (which will be used for pose estimation)
    '''
    A = T[:,:2]
    b = T[:,2]
    kpts_new = np.linalg.solve(A,kpts*4 - b[:,np.newaxis])
    lam_new = []
    for lam in lams:
        lam_new.append( (A.T @ lam @ A)/16 )
    return kpts_new, np.stack(lam_new)

parser = argparse.ArgumentParser()
parser.add_argument('--score_type', action='store', type=str)
parser.add_argument('--eps', type=int)
parser.add_argument('--do_frcnn', action='store_true')
parser.add_argument('--save_fig', action='store_true')

args = parser.parse_args()

score_type = args.score_type
eps = args.eps
eps = eps / 100.0
save_fig = args.save_fig
do_frcnn = args.do_frcnn

print(f'nonconformity function: {score_type}, epsilon: {eps}, save_fig: {save_fig}.')

# Load dataset 
dataset_name = 'lmo-org' # this is the full lmo dataset containing 1214 images
root         = './keypoint/data/bop'
num_classes  = {'lmo':8, 'lmo-org':8} 
device       = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset      = BOPDataset(root, dataset_name, split='test', return_coco=True)
dataset._set_kpts_info()

if do_frcnn:
    # Load Faster-RCNN detector
    detector_trainsform = T.ToTensor()
    state_dict = torch.load('keypoint/data/detect_checkpoints/d{}.pt'.format('lmo'), map_location=device)['frcnn']
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
n_objs = len(idx2obj)
poseEval = Pose2DEval()

img_result_dir = './keypoint/data/bop/lmo-org/test/000002/icp_results'

fname = f'calibration_scores_{score_type}_lmo.pkl'
if do_frcnn:
    fname = f'calibration_scores_{score_type}_lmo_frcnn.pkl'
# Compute quantiles
with open(fname, 'rb') as f:
    obj_scores = pickle.load(f)
obj_qs = []
for i in range(n_objs):
    scores = obj_scores[i]
    n      = np.size(scores)
    idx    = np.int64( np.floor( (n+1) * eps ) )
    scores_sort = scores[np.flip(np.argsort(scores))]
    obj_qs.append(scores_sort[idx-1])
obj_qs = np.array(obj_qs)

# Perform Conformal prediction
obj_kpts = [[] for i in range(n_objs)]
obj_lams = [[] for i in range(n_objs)]
obj_imgs = [[] for i in range(n_objs)]

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
        pd_boxes, pd_labels = one_each(pred, thresh=0)
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
        gt_kpt_homo = np.concatenate((gt_kpt,np.ones((gt_kpt.shape[0],1))),axis=1)
        input_crop  = {'image':image, 'bb':box, 'keypoints':gt_kpt_homo} 
        input_crop  = kpts_transform(input_crop)
        gt_kpt_crop = input_crop['keypoints'][:,:2].numpy() * (64/256) # the heatmap is size (64,64), rescale from 256 to 64
        # affine transformation between original kpt loc and that in heatmap
        affineT     = transform_list[0].crop_augment(box) 
        
        with torch.no_grad():
            batch = input_crop['image'][None].to(device)
            output = kpts_detector(batch)
            output = output[-1].cpu()
        
        kpt_start       = dataset.obj2kptid[obj][0]
        kpt_end         = dataset.obj2kptid[obj][1]
        heatmaps_pred   = torch.squeeze(output[[0], kpt_start:kpt_end, :, :])

        # output inductive conformal prediction set
        kpts = []
        lams = []
        icp_sets = []
        for j in range(kpt_start,kpt_end):
            if score_type == "ball":
                center, radius = icp(
                    torch.squeeze(heatmaps_pred[j-kpt_start,:]).numpy(),
                    obj_qs[obj2idx[obj]],
                    type=score_type)

                lam = np.eye(2) / (radius**2)
                kpts.append(center) # center
                lams.append(lam) # information matrix
                icp_sets.append((center,radius))

            elif score_type == "ellipse":
                center, lam = icp(
                    torch.squeeze(heatmaps_pred[j-kpt_start,:]).numpy(),
                    obj_qs[obj2idx[obj]],type=score_type)
                kpts.append(center)
                lams.append(lam)      
                icp_sets.append((center,lam))  

            else:
                raise RuntimeError('Unknown score type for ICP.')
        
        if save_fig:
            dir_path = "{:s}/{:.2f}/{:s}".format(img_result_dir,eps,score_type)
            os.makedirs(dir_path,exist_ok=True)
            fname = "{:s}/{:06d}_{:06d}_{:02d}.pdf".format(dir_path,i,meta['im_id'],obj)
            if do_frcnn:
                fname = "{:s}/{:06d}_{:06d}_{:02d}_frcnn.pdf".format(dir_path,i,meta['im_id'],obj)
            # plot
            img_disp = cv2.resize((input_crop['image'].permute(1, 2, 0).numpy()) / 2.0 + 0.5,(64,64))
            if score_type == "ball":
                fig = draw_icp_ball(img_disp,heatmaps_pred.numpy(),gt_kpt_crop,icp_sets,fname=fname,show=False)
            elif score_type == "ellipse":
                fig = draw_icp_ellipse(img_disp,heatmaps_pred.numpy(),gt_kpt_crop,icp_sets,fname=fname,show=False)
            plt.close(fig)

        kpts = np.stack(kpts,axis=1)
        # convert the keypoints coordinates to the original image space and save
        kpts_new, lams_new = heatmap2org(kpts,lams,affineT)
        obj_kpts[obj2idx[obj]].append(kpts_new)
        obj_lams[obj2idx[obj]].append(lams_new)
        obj_imgs[obj2idx[obj]].append(i)

# save the keypoint prediction sets
data = {"kpts": obj_kpts,
        "lams": obj_lams,
        "imgs": obj_imgs}
fname = "icp_sets_{:s}_{:.2f}.pkl".format(score_type,eps)
if do_frcnn:
    fname = "icp_sets_{:s}_{:.2f}_frcnn.pkl".format(score_type,eps)
with open(fname, 'wb') as f:
    pickle.dump(data, f)