from __future__ import print_function, unicode_literals
import matplotlib.pyplot as plt
from data.FreiHAND.fh_utils import *
import utils.utils as utils
from torch.utils.data import Dataset
import random

# base_path='C:\\Users\\lahir\\Downloads\\FreiHAND_pub_v2\\'
# db_data_anno = list(load_db_annotation(base_path, 'training'))


def show_training_samples(base_path, version, num2show=None, render_mano=False):
    # if render_mano:
    #     from utils.model import HandModel, recover_root, get_focal_pp, split_theta

    if num2show == -1:
        num2show = db_size('training') # show all

    # load annotations
    db_data_anno = list(load_db_annotation(base_path, 'training'))

    # iterate over all samples
    for idx in range(db_size('training')):
        if idx >= num2show:
            break

        # load image and mask
        img = read_img(idx, base_path, 'training', version)
        msk = read_msk(idx, base_path)

        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)

        # render an image of the shape
        msk_rendered = None
        # if render_mano:
        #     # split mano parameters
        #     poses, shapes, uv_root, scale = split_theta(mano)
        #     focal, pp = get_focal_pp(K)
        #     xyz_root = recover_root(uv_root, scale, focal, pp)

        #     # set up the hand model and feed hand parameters
        #     renderer = HandModel(use_mean_pca=False, use_mean_pose=True)
        #     renderer.pose_by_root(xyz_root[0], poses[0], shapes[0])
        #     msk_rendered = renderer.render(K, img_shape=img.shape[:2])

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img)
        ax2.imshow(msk if msk_rendered is None else msk_rendered)
        plot_hand(ax1, uv, order='uv')
        plot_hand(ax2, uv, order='uv')
        ax1.axis('off')
        ax2.axis('off')
        plt.show()


def show_eval_samples(base_path, num2show=None):
    if num2show == -1:
        num2show = db_size('evaluation') # show all

    for idx in  range(db_size('evaluation')):
        if idx >= num2show:
            break

        # load image only, because for the evaluation set there is no mask
        img = read_img(idx, base_path, 'evaluation')

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)
        ax1.axis('off')
        plt.show()

# show_training_samples('C:\\Users\\lahir\\Downloads\\FreiHAND_pub_v2_training\\','hom',num2show=10)
'''
ranges of the intrinsic matrix
min :
array([[190.78378074,   0.        , 112.        ],
       [  0.        , 190.74944027, 112.        ],
       [  0.        ,   0.        ,   1.        ]])

max:
array([[1214.18, 0,       112],
       [0,       1213.68, 112],
       [0,       0      , 1]])

mean:
array([[483.79409933,   0.        , 112.        ],
       [  0.        , 483.81574158, 112.        ],
       [  0.        ,   0.        ,   1.        ]])

'''
class FreiHAND(Dataset):
    def __init__(self,conf,mode):
        self.conf=conf
        # load annotations
        self.mode=mode
        self.base_path=os.path.join(conf.datasets.freihand.base_path,f"FreiHAND_pub_v2_{self.mode}")
        if conf.datasets.freihand.unet_annot:
            self.db_data_anno,self.unet_annot = load_db_annotation(self.base_path,mode,self.conf.datasets.freihand.unet_annot)
            self.db_data_anno=list(self.db_data_anno)
        else:
            self.db_data_anno = list(load_db_annotation(self.base_path,self.mode))

        #load the coco split files
        if self.mode=='evaluation':
            coco_path=os.path.join(self.base_path,'freihand_eval_coco.json')
            #the only version that makes sense for the evaluation set is the 0th choice
            self.version=sample_version.valid_options()[0]
            self.get_mask=False
        else:
            coco_path=os.path.join(self.base_path,'freihand_train_coco.json')
            self.version=conf.datasets.freihand.version
            self.get_mask=conf.datasets.freihand.get_mask

        with open(coco_path, 'r') as fi:
            self.coco = json.load(fi)['images']
        
        print(f'length of the dataset : {len(list(self.db_data_anno))}')
        print(f'image version used : {self.version}')
    
    def get_unet_annot(self,idx):
        version=self.version
        set_name=self.mode

        if version==-1:
            versions=sample_version.valid_options()
            version=random.choice(versions)

        if version is None:
            version = sample_version.gs

        if set_name == 'evaluation':
            assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

        unet_idx=sample_version.map_id(idx, version)
        return self.unet_annot[unet_idx]


    def __len__(self):
        return len(list(self.db_data_anno))

    def __getitem__(self, idx): 
        #randomly select one of the versions to get data
        # if self.version==-1:
        #     versions=sample_version.valid_options()
        #     version=random.choice(versions)
        # else:
        #     version=self.version
        
        #chose the index based on coco file
        coco_idx=self.coco[idx]['db_idx']

        # annotation for this frame
        K, mano, xyz = self.db_data_anno[coco_idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)
        values={}
        if self.conf.datasets.freihand.get_image:
            img_pth=os.path.join(self.base_path,self.coco[idx]['file_name'])
            img=read_img_from_pth(img_pth)
            # img = read_img(idx, self.base_path,self.mode, version)
            values['img']=img
        if self.conf.datasets.freihand.get_mask:
            values['msk']=msk
        values['K']=K
        values['xyz']=xyz
        values['uv']=uv
        if self.get_mask:
            msk = read_msk(coco_idx, self.base_path)
            values['msk']=msk
        dists=utils.get_euclidian_dist_pt(xyz)
        values['dists']=dists
        xyz_=np.moveaxis(xyz,1,0)
        xyz_=np.expand_dims(xyz_,-1)
        values['hand_dim']=utils.get_hand_dims(xyz_,self.conf)[:,0]
        #get 2D keypoint annotations generated with unet
        if self.conf.datasets.freihand.unet_annot:
            annot=self.get_unet_annot(idx)
            values['unet_annot']=np.array(annot['keypoints'])
        return values
    
def get_dist(dataset):
    import numpy as np
    import matplotlib.pyplot as plt

    dists=np.empty(0)
    for i in range(len(dataset)):
        try:
            values=dataset[i]
            dists=np.concatenate((dists,utils.get_euclidian_dist_pt(values['xyz'])),axis=0) 
            print(f'idx = {i}')
        except Exception as e:
            print(e)
    #plot dist histogram
    plt.hist(dists)
    plt.show()
    return dists

def get_intrinsic(dataset):
    import numpy as np
    import matplotlib.pyplot as plt

    k_mats=np.empty(0,3,3)
    for i in range(len(dataset)):
        try:
            values=dataset[i]['K']
            k_mats=np.concatenate((k_mats,values),axis=0) 
            print(f'idx = {i}')
        except Exception as e:
            print(e)
    return k_mats



    
# base_path='C:\\Users\\lahir\\Downloads\\'
# dataset=FreiHAND(base_path,mode='training')
# values=dataset[1000]
# show_img_kpts(values)



