import hydra
from omegaconf import DictConfig, OmegaConf
import data.FreiHAND.dataset as dataset
import utils.utils as utils
from utils.tasks import Trainer,Tester
# import wandb
# wandb.login()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(conf : DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))

    trainer=Trainer(conf)
    trainer._make_model()
    trainer._make_dataloader()

    tester=Tester(conf)
    tester._make_dataloader()

    tester.set_model(trainer.model)
    rmse_error=tester.evaluate()

    for epoch in range(conf.train.n_epochs):
        print(f'Starting epoch {epoch}')
        trainer.train_epoch()        
        if (epoch+1)%conf.train.eval_freq==0:
            tester.set_model(trainer.model)
            rmse_error=tester.evaluate()
            print(f"evaluation RMSE = {rmse_error}")
            
    # rmse_error=random.randint(1,300) 
    # print(f"evaluation RMSE = {rmse_error}")
    return rmse_error
        
if __name__ == "__main__":
    train()

# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def train(conf : DictConfig) -> None:
#     global c
#     c=conf
# train()

# tester=Tester(c)
# tester._make_dataloader()

# trainer=Trainer(c)
# trainer._make_dataloader()

# d=iter(trainer.data_loader).next()
# unet=d['unet_annot'][0,:,:]
# gt_kp= d['uv'][0,:,:]
# import numpy as np
# import matplotlib.pyplot as plt
# img=d['img']
# plt.imshow(img[0,:,:,:])
# plt.scatter( unet[:,0],unet[:,1],c='b')
# plt.scatter( gt_kp[:,0],gt_kp[:,1],c='r')
# plt.show()


# def projectPoints(xyz, K):
#     """ Project 3D coordinates into image space. """
#     xyz = np.array(xyz)
#     K = np.array(K)
#     uv = np.matmul(K, xyz.T).T
#     return uv[:, :2] / uv[:, -1:]


# import skimage.io as io
# import matplotlib.pyplot as plt
# import json

# imgpth=r'C:\Users\lahir\Downloads\FreiHAND_pub_v2_training\training\rgb\00032561.jpg'
# gt_xyz_pth=r'C:\Users\lahir\Downloads\FreiHAND_pub_v2_training\training_xyz.json'
# k_pth=r'C:\Users\lahir\Downloads\FreiHAND_pub_v2_training\training_K.json'
# unet_pth=r'C:\Users\lahir\Downloads\FreiHAND_pub_v2_training\hrnet_output_on_trainset.json'
# coco_train_pth=r'C:\Users\lahir\Downloads\FreiHAND_pub_v2_training\freihand_train_coco.json'


# with open(k_pth, 'r') as fi:
#     k =json.load(fi)

# with open(k_pth, 'r') as fi:
#     k = np.array(json.load(fi)[1])
# with open(gt_xyz_pth, 'r') as fi:
#     gt_xyz = np.array(json.load(fi)[1])
# with open(unet_pth, 'r') as fi:
#     unet_uv = np.array(json.load(fi)[5]['keypoints'])
# with open(coco_train_pth, 'r') as fi:
#     coco = json.load(fi)

# coco['images'][5]

# uv=projectPoints(gt_xyz,k)

# img=io.imread(imgpth)
# plt.imshow(img)
# plt.scatter(uv[:,0],uv[:,1])
# plt.scatter(unet_uv[:,0],unet_uv[:,1])
# plt.show()



# # import torch
# # kypts=torch.permute(d['xyz'],(2,1,0))

# # hand_dims=utils.get_hand_dims(kypts,c)




































