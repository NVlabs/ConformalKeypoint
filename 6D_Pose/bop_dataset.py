import os
import json
import glob
import numpy as np
import torch
from PIL import Image


from bop_toolkit_lib.dataset_params import *
from bop_toolkit_lib.inout import load_im, load_json

"""
Dataset class that can be used to load BOP data to train detection/keypoint network
The backbone is basically done. Still need to implement:
1. enable data augmentation
2. output crop with uniform dimention (e.g. 256x256)
3. adjust keypoints to augmented, cropped output
4. ....
"""

class BOPDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_root, dataset_name, split='train', transform=None,
                 vis_thresh=0.2, return_keypoints=False, return_coco=False,
                 valid_objid = None):

        if split is 'train':
            p = get_split_params(datasets_root, dataset_name=dataset_name, 
                                 split=split, split_type='pbr')
        else:
            p = get_split_params(datasets_root, dataset_name=dataset_name, 
                                 split=split, split_type=None)

        present_scene_ids = get_present_scene_ids(p)
        p['scene_ids'] = present_scene_ids

        self.split_path = p['split_path']
        self.img_tpath = p['rgb_tpath']
        self.gt_tpath  = p['scene_gt_tpath']
        self.box_tpath = p['scene_gt_info_tpath']
        self.cam_tpah  = p['scene_camera_tpath']
        self.valid_objid = valid_objid

        self.transform = transform
        self.dataset_name = dataset_name
        self.vis_thresh = vis_thresh
        self.return_coco = return_coco
        self.return_keypoints = return_keypoints

        self.db = self._get_db(scene_ids = present_scene_ids)
        
    def __len__(self):
        return len(self.db)
    

    def __getitem__(self, index):
        rec = self.db[index]
        
        im = self.load_img(rec['imgpath'])
        sample = {}

        # return detection related labels (for torchvision)
        if self.return_coco:
            sample['image'] = im
            sample['boxes'] = rec['boxes'] # (x0,y0,x1,y1) format
            sample['labels'] = rec['labels']

            if self.transform is not None:
                sample = self.transform(sample)

            return sample #sample['image'], {k:v for k,v in sample.items() if k!='image'}

        # return keypoint related labels
        if self.return_keypoints:
            sample['image'] = im
            sample['bb'] = rec['bbox']  # (x0,y0,w,h) format
            sample['keypoints'] = rec['keypoints']

            if self.transform is not None:
                sample = self.transform(sample)

            return sample

    
    def _get_db(self, scene_ids):
        # Loading in the info is slow
        db = []
        image_id = -1   #unique image id for coco-related tasked
        valid_objid = set()

        for i in scene_ids:
            gt_path = self.gt_tpath.format(scene_id=i)
            box_path = self.box_tpath.format(scene_id=i)
            cam_path = self.cam_tpah.format(scene_id=i)
            
            scene_gt  = load_json(gt_path, keys_to_int=True)
            scene_box = load_json(box_path, keys_to_int=True)
            scene_cam = load_json(cam_path, keys_to_int=True)

            img_folder = os.path.join(self.split_path, '{scene_id:06d}/rgb/*').format(scene_id=i)
            img_ids = [int(os.path.basename(d).split('.')[0]) for d in glob.glob(img_folder)]
            img_ids = sorted(img_ids)
            
            for j in img_ids:
                image_id += 1
                im_gt  = scene_gt[j]
                im_box = scene_box[j]
                im_cam = scene_cam[j]
                im_path = self.img_tpath.format(scene_id=i, im_id=j)
                
                for k in range(len(im_box)):
                    if im_box[k]['visib_fract'] > self.vis_thresh:
                        bbox = im_box[k]['bbox_obj']
                        obj_id = im_gt[k]['obj_id']
                        R = im_gt[k]['cam_R_m2c']
                        t = im_gt[k]['cam_t_m2c']
                        K = im_cam['cam_K']

                        # only include obj we want (for lmo dataset)
                        if self.valid_objid is not None:
                            if obj_id not in self.valid_objid:
                                continue
                        
                        rec = {'imgpath': im_path,
                               'bbox': bbox,
                               'obj_id': obj_id,
                               'R': np.array(R).reshape(3,3),
                               't': np.array(t),
                               'K': np.array(K).reshape(3,3), 
                               'image_id': image_id,
                               'scene_id':i,
                               'im_id':j}
                        
                        db.append(rec)
                        valid_objid.add(obj_id)
        

        # Map objid to labels
        self.obj2idx = {}
        for i, idx in enumerate(valid_objid):
            self.obj2idx[idx] = i 

        if self.valid_objid is None:
            self.valid_objid = valid_objid


        # Append new entry for keypoint or detection related task
        if self.return_coco is True:
            db = self._get_detection_db(db)

        if self.return_keypoints is True:
            db = self._get_keypoint_db(db)

            
        return db



    def _get_detection_db(self, db):
        with open('kpts3d.json', 'r') as infile:
            dataset = self.dataset_name
            if dataset == "lmo-org":
                dataset = 'lmo'
            kpts3d = json.load(infile)[dataset]

        detection_db = {}

        imgid = -1
        refpath = ' '
        items = []
        for rec in db:
            imgpath = rec['imgpath']
            if imgpath != refpath:
                refpath = imgpath
                scene_id = rec['scene_id']
                im_id = rec['im_id']
                K = rec['K']
                imgid += 1

                boxes = []
                labels = []
                Rs = []
                ts = []
                obj_ids = []
                keypoints = []


            bb = rec['bbox']
            boxes.append([bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]])
            obj_ids.append(rec['obj_id'])
            labels.append(self.obj2idx[rec['obj_id']] + 1)
            Rs.append(rec['R'])
            ts.append(rec['t'])

            obj_id = rec['obj_id']
            kpts = np.array(kpts3d[str(obj_id)])
            kpts_3d = (rec['R']@kpts.T).T + rec['t']
            kpts_2d = (rec['K']@kpts_3d.T).T
            kpts_2d = kpts_2d[:,:2]/kpts_2d[:,[2]]
            keypoints.append(kpts_2d)


            detection_db[imgid] = {# one per scene
                                   'imgpath': refpath,
                                   'scene_id': scene_id,
                                   'im_id': im_id,
                                   'K': K,

                                   # list: one per obj in the scene
                                   'boxes': boxes,
                                   'labels': labels,
                                   'obj_ids': obj_ids,
                                   'Rs': Rs,
                                   'ts': ts,
                                   'keypoints': keypoints
                                   }

        detection_db = [v for k,v in detection_db.items()]
        return detection_db
                    

    def _get_keypoint_db(self, db):
        with open('kpts3d.json', 'r') as infile:
            dataset = self.dataset_name
            if dataset == "lmo-org":
                dataset = 'lmo'
            kpts3d = json.load(infile)[dataset]

        n_kpts = []
        obj2idx = {}
        idx = 0
        for obj_id in kpts3d:
            if self.valid_objid is None:
                n_kpts.append(len(kpts3d[obj_id]))
                obj2idx[int(obj_id)] = idx
                idx += 1
            else:
                if int(obj_id) in self.valid_objid:
                    n_kpts.append(len(kpts3d[obj_id]))
                    obj2idx[int(obj_id)] = idx
                    idx += 1
        n_kpts = [sum(n_kpts[:i]) for i in range(len(n_kpts)+1)]

        self.obj2idx = obj2idx
        self.obj2kptid = {i:(n_kpts[k], n_kpts[k+1]) for i, k in obj2idx.items()}
        self.n_kpts = n_kpts[-1]

        for rec in db:
            obj_id = rec['obj_id']
            kpts = np.array(kpts3d[str(obj_id)])
            kpts_3d = (rec['R']@kpts.T).T + rec['t']
            kpts_2d = (rec['K']@kpts_3d.T).T
            kpts_2d = kpts_2d[:,:2]/kpts_2d[:,[2]]
            bop_kpts = self._get_bop_keypoints(obj2idx[obj_id], kpts_2d, n_kpts)

            rec['keypoints'] = bop_kpts  

        return db


    def _set_kpts_info(self):
        with open('kpts3d.json', 'r') as infile:
            dataset = self.dataset_name
            if dataset == "lmo-org":
                dataset = 'lmo'
            kpts3d = json.load(infile)[dataset]

        n_kpts = []
        obj2idx = {}
        idx = 0
        for obj_id in kpts3d:
            if self.valid_objid is None:
                n_kpts.append(len(kpts3d[obj_id]))
                obj2idx[int(obj_id)] = idx
                idx += 1
            else:
                if int(obj_id) in self.valid_objid:
                    n_kpts.append(len(kpts3d[obj_id]))
                    obj2idx[int(obj_id)] = idx
                    idx += 1
        n_kpts = [sum(n_kpts[:i]) for i in range(len(n_kpts)+1)]

        self.obj2idx = obj2idx
        self.obj2kptid = {i:(n_kpts[k], n_kpts[k+1]) for i, k in obj2idx.items()}
        self.n_kpts = n_kpts[-1]
        self.kpts3d = kpts3d
        return

    def _get_bop_keypoints(self, idx, kpts, n_kpts):
        keypoints = np.zeros([n_kpts[-1], 3])
        keypoints[n_kpts[idx] : n_kpts[idx+1], :2] = kpts
        keypoints[n_kpts[idx] : n_kpts[idx+1],  2] = 1

        return keypoints



    def _transform_bbox(self, bbox):
        x, y, w, h = [np.maximum(0,i) for i in bbox]
        bb = np.zeros([4,2])
        bb[0,:] = [x, y]
        bb[1,:] = [x+w, y]
        bb[2,:] = [x, y+h]
        bb[3,:] = [x+w, y+h]
        return bb


    def load_img(self, img_path):
        try:
            im = Image.open(img_path).convert(mode='RGB')
        except FileNotFoundError:
            alter_path = img_path.replace('.jpg', '.png')
            im = Image.open(alter_path).convert(mode='RGB')

        return im



    