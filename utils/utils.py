import numpy as np

# joint_idx={
#     'root':0,
#     'little': {'mcp':1, 'pip':2, 'dip':3, 'tip':4},
#     'ring': {'mcp':5, 'pip':6, 'dip':7, 'tip':8},
#     'middle': {'mcp':9, 'pip':10, 'dip':11, 'tip':12},
#     'index': {'mcp':13, 'pip':14, 'dip':15, 'tip':16},
#     'thumb': {'mcp':17, 'pip':18, 'dip':19, 'tip':20}
# }

#subtract the root coordinate of each sample from the joint coordinates
# joint_coords : np.array of shape (3, n_joints, n_samples)
# root_n : index of the root joint
def joint_normalize(joint_coords,root_n=0):
    palm_centers=np.expand_dims(joint_coords[:,root_n,:],axis=1)
    palm_centers=np.repeat(palm_centers,repeats=joint_coords.shape[1] ,axis=1)
    joint_coords_norm=joint_coords-palm_centers
    return joint_coords_norm


def rearrange_array(a):
    assert len(a.shape)==2 , "input must be 2D"
    assert a.shape[0]==3 or a.shape[1]==3 , "one of the dimentions of the input must be 3 (for XYZ)"
    if a.shape[1]==3:
        a=np.moveaxis(a,1,0)
    return a

#get dist between two points
def get_euclidian_dist(a1,a2):
    a1=rearrange_array(a1)
    a2=rearrange_array(a2)
    return np.sqrt(np.sum(np.square(a1-a2),axis=0))

#get dist to a point from origin
def get_euclidian_dist_pt(a):
    a=rearrange_array(a)
    return np.sqrt(np.sum(np.square(a),axis=0))

'''
get euclidian distances between consecetive hand joints
for 21 joints there will be 20 distances

items returned:
0: little_mcp - root
1: little_pip - little_mcp
2: little_dip - little_pip
3: little_tip - little_dip
4: ring_mcp - root
.....

kypts: np array of shape (3,n_joints,n_samples)

'''
def get_hand_dims(kypts,conf):
    joint_idx=conf.datasets[conf.dataset].joint_idx
    hand_dim=np.empty((0, kypts.shape[-1]))
    for key in joint_idx.keys():
        if key=='root': continue        

        #for each finger
        rel=(get_euclidian_dist(kypts[:,joint_idx[key]['mcp'],:],kypts[:,joint_idx['root'],:]))
        # print(f'root {rel}')
        rel=np.expand_dims(rel,axis=0)
        hand_dim=np.append(hand_dim,rel,axis=0)

        #MCP to pip distance
        rel=(get_euclidian_dist(kypts[:,joint_idx[key]['pip'],:],kypts[:,joint_idx[key]['mcp'],:]))
        # print(f'pip mcp {rel}')
        rel=np.expand_dims(rel,axis=0)
        hand_dim=np.append(hand_dim,rel,axis=0)

        #pip to dip distance
        rel=(get_euclidian_dist(kypts[:,joint_idx[key]['dip'],:],kypts[:,joint_idx[key]['pip'],:]))
        # print(f'dip pip {rel}')
        rel=np.expand_dims(rel,axis=0)
        hand_dim=np.append(hand_dim,rel,axis=0)

        #dip to tip distance
        rel=(get_euclidian_dist(kypts[:,joint_idx[key]['tip'],:],kypts[:,joint_idx[key]['dip'],:]))
        # print(f'dip tip {rel}')
        rel=np.expand_dims(rel,axis=0)
        hand_dim=np.append(hand_dim,rel,axis=0)

    return hand_dim


#get distance (depth of XYZ points) of a given dataset
def get_dists(dataset):
    dists=np.empty(0)
    for i in range(len(dataset)):
        try:
            values=dataset[i]
            dists=np.concatenate((dists,get_euclidian_dist_pt(values['xyz'])),axis=0) 
        except Exception as e:
            print(e)
    return dists

#create dataloader from dataset
import data.FreiHAND.dataset as dataset
from torch.utils.data import DataLoader

def get_dataloaders(conf):
    dloaders={}
    if 'training' in conf.mode:
        d_tr=dataset.FreiHAND(conf,mode='training')
        train_dataloader = DataLoader(d_tr, batch_size=conf.datasets.freihand.bs, shuffle=True)
        dloaders['train']=train_dataloader
    if 'evaluation' in conf.mode:
        d_eval=dataset.FreiHAND(conf,mode='evaluation')
        eval_dataloader = DataLoader(d_eval, batch_size=1, shuffle=True)
        dloaders['eval']=d_eval
    return dloaders

#count the number of parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#get all intrinsic matrices in the dataset
def get_intrinsic_mats(dataloader):
    import numpy as np
    k_mats=np.empty((0,3,3))
    for i, data in enumerate(dataloader, 0):
        values=data['K'].numpy()
        k_mats=np.concatenate((k_mats,values),axis=0) 
    return k_mats

#get all relative coordinates from the dataloader
def get_rel_coords(dataloader):
    import torch
    coords=torch.empty((0,21,2))
    for i,inputs in enumerate(dataloader):
        roots=torch.repeat_interleave(torch.unsqueeze(inputs['uv'][:,0,:],dim=1),repeats=inputs['uv'].shape[1],dim=1)
        fx=inputs['K'][:,0,0]
        fy=inputs['K'][:,1,1]
        fx=torch.unsqueeze(fx,dim=1).unsqueeze(dim=2)
        fx=torch.repeat_interleave(fx,repeats=inputs['uv'].shape[1],dim=1)
        fy=torch.unsqueeze(fy,dim=1).unsqueeze(dim=2)
        fy=torch.repeat_interleave(fy,repeats=inputs['uv'].shape[1],dim=1)
        f=torch.concat((fx,fy),dim=-1)
        rel_coord=(inputs['uv']-roots)/f
        coords=torch.cat((coords,rel_coord),dim=0)
    return coords
