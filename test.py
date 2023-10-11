# #assume we have a cropped hand image
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# impth=r'C:\Users\lahir\Downloads\hands\original\cropped\Rhand_up.jpg'
# img=cv2.imread(impth, cv2.IMREAD_COLOR)
# cv2.imshow('name',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# impth=r'D:\data\RTB dataset\stereo hand pose dataset\images\B1Counting\SK_color_0.png'
# img=cv2.imread(impth, cv2.IMREAD_COLOR)
# cv2.imshow('name',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# SK_depth_intrinsic=np.array([[475.62768,0,336.41179],[0,474.77709,238.77962],[0,0,1]])
# SK_color_intrinsic=np.array([[607.92271,0,314.78337],[0,607.88192,236.42484],[0,0,1]])



# #read the labels
# labelfile='D:\data\RTB dataset\stereo hand pose dataset\labels\B1Counting_SK.mat'
# from scipy.io import loadmat 
# labels=loadmat(labelfile)['handPara']
# joint_coords=labels[:,:,0]
# #project to camera image plane
# proj=np.matmul(SK_color_intrinsic,joint_coords)
# #convert to non-homogenous from homogenous
# proj=proj/proj[-1,:]
# proj=proj[:2,:]

# im = plt.imread(impth)
# implot = plt.imshow(im)
# plt.scatter(proj[0,:],proj[1,:])
# plt.show()



# joint_coords_xy=joint_coords[:2,:]





# plt.figure()
# plt.scatter(proj[0,:],proj[1,:])
# plt.show()


# #subtract the root coordinate of each sample from the joint coordinates
# # joint_coords : np.array of shape (3, n_joints, n_samples)
# # root_n : index of the root joint
# def joint_normalize(joint_coords,root_n=0):
#     palm_centers=np.expand_dims(joint_coords[:,root_n,:],axis=1)
#     palm_centers=np.repeat(palm_centers,repeats=labels.shape[1] ,axis=1)
#     joint_coords_norm=joint_coords-palm_centers
#     return joint_coords_norm

# #get hand dimentions: euclidian distances of bones of the hand
# joint_idx={
#     'root':0,
#     'little': {'mcp':1, 'pip':2, 'dip':3, 'tip':4},
#     'ring': {'mcp':5, 'pip':6, 'dip':7, 'tip':8},
#     'middle': {'mcp':9, 'pip':10, 'dip':11, 'tip':12},
#     'index': {'mcp':13, 'pip':14, 'dip':15, 'tip':16},
#     'thumb': {'mcp':17, 'pip':18, 'dip':19, 'tip':20}
# }

# def get_euclidian_dist(a1,a2):
#     return np.sqrt(np.sum(np.square(a1-a2),axis=0))


# '''
# get euclidian distances between consecetive hand joints
# for 21 joints there will be 20 distances

# items returned:
# 0: little_mcp - root
# 1: little_pip - root
# 2: little_dip - root
# 3: little_tip - root
# 4: ring_mcp - root
# .....

# kypts: np array of shape (3,n_joints,n_samples)

# '''
# def get_hand_dims(kypts):
#     hand_dim=np.empty((0, kypts.shape[-1]))
#     for key in joint_idx.keys():
#         if key=='root': continue

#         rel=(get_euclidian_dist(kypts[:,joint_idx[key]['mcp'],:],kypts[:,joint_idx['root'],:]))
#         rel=np.expand_dims(rel,axis=0)
#         hand_dim=np.append(hand_dim,rel,axis=0)
#         for subkey in joint_idx[key].keys():
#             if subkey=='mcp':  continue

#             rel=(get_euclidian_dist(kypts[:,joint_idx[key][subkey],:],kypts[:,joint_idx[key]['mcp'],:]))
#             rel=np.expand_dims(rel,axis=0)
#             hand_dim=np.append(hand_dim,rel,axis=0)
#     return hand_dim

import hydra
from omegaconf import DictConfig, OmegaConf
import data.FreiHAND.dataset as dataset
import utils.utils as utils
from utils.tasks import Trainer,Tester

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))
    global c
    c=conf

    trainer=Trainer(c)
    trainer._make_model()
    trainer._make_dataloader()

    tester=Tester(c)
    tester._make_dataloader()

    tester.set_model(trainer.model)
    tester.evaluate()

    for epoch in range(c.train.n_epochs):
        print(f'Starting epoch {epoch}')
        for itr,inputs in enumerate(trainer.data_loader):
            trainer.optimizer.zero_grad()
            model_out=trainer.model(inputs)
            loss=trainer.loss_func(model_out.double(),inputs['dists'].double())
            loss.backward()
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            # print(itr/len(trainer.data_loader))
        
        if (epoch+1)%conf.train.eval_freq==0:
            tester.set_model(trainer.model)
            tester.evaluate()
        

if __name__ == "__main__":
    main()

# dl=utils.get_dataloaders(c)['train']



# print('end')











    

# import torch
# import torch.nn as nn
# encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
# src = torch.rand(10, 32, 512)
# out = encoder_layer(src)










# import json
# import numpy as np

# coords=np.empty((0,3))

# path=r'C:\Users\lahir\Downloads\FreiHAND_pub_v2\freihand_train_data.json'
# f = open(path)
# data=json.load(f)

# for key in data.keys():
#     c=np.expand_dims(np.array(data[key]['joint_3d'][0]),axis=0)
#     coords=np.concatenate((coords,c),axis=0)

# dists=np.sqrt(np.sum(np.square(coords),axis=1))
# np.min(dists),np.max(dists)






























