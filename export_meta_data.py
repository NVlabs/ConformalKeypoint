from keypoint.bop_dataset import BOPDataset
import json


dataset_name = 'lmo-org' # this is the full lmo dataset containing 1214 images
root         = './keypoint/data/bop'
num_classes  = {'lmo':8, 'lmo-org':8} 
dataset      = BOPDataset(root, dataset_name, split='test', return_coco=True)
dataset._set_kpts_info()

n_kpts  = dataset.n_kpts
n_smps  = len(dataset)
obj2idx = dataset.obj2idx
idx2obj = {v:k for k,v in obj2idx.items()}
lab2obj = {v+1:k for k,v in obj2idx.items()}
n_objs = len(idx2obj)

with open('keypoint/kpts3d.json', 'r') as infile:
    if dataset_name == 'lmo-org':
        kpts3d = json.load(infile)['lmo']
    else:
        kpts3d = json.load(infile)[dataset_name]

kpts3d = [kpts3d[str(obj)] for obj in obj2idx.keys()]


with open('keypoint/data/bop/lmo-org/test/000002/scene_camera.json', 'r') as f:
    cam_params_all = json.load(f)
    
cam_params:dict = cam_params_all['0']
for k in range(len(cam_params_all)):
    assert cam_params.items() == cam_params_all[str(k)].items(), "Camera parameters incompatible!"
    

with open("meta_data.json", 'w') as f:
    json.dump({"kpts3d":kpts3d, "cam_params":cam_params}, f)