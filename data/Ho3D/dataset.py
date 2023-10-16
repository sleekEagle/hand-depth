import utils as utils
import os
from torch.utils.data import Dataset
import random
import numpy as np

class Ho3D(Dataset):
    def __init__(self):
        base_path='D:\data\HO3D_v2\HO3D_v2\HO3D_v2'
        set_name='train'
        self.seq_len=7
        self.set_path=os.path.join(base_path,set_name)

        #get the number of files in each dir
        dir_sizes_={}
        dir_names=os.listdir(self.set_path)
        for d in dir_names:
            n_files=len(os.listdir(os.path.join(base_path,set_name,d,'rgb')))
            dir_sizes_[d]=n_files
        self.dir_sizes=dir_sizes_

        #read the file names
        with open(os.path.join(base_path, set_name+'.txt')) as f:
            file_list = f.readlines()
        self.file_list = [f.strip() for f in file_list]
        assert len(file_list) == utils.db_size(set_name, 'v2'), '%s.txt is not accurate. Aborting'%set_name

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        print(f'idx={idx}')
        file_name=self.file_list[idx]
        file_num=int(file_name.split('/')[-1])
        dir_name=file_name.split('/')[0]
        dir_size=self.dir_sizes[dir_name]

        if file_num>dir_size-self.seq_len:
            diff=file_num-(dir_size-self.seq_len)
            start_pos=file_num-diff
        else:
            start_pos=file_num

        #extract sequences of images
        img_seq=np.empty((0,480,640,3))
        for pos in range(start_pos,start_pos+self.seq_len):
            img=utils.read_RGB_img(self.set_path,dir_name,f"{pos:0{4}}")
            img=np.expand_dims(img,axis=0)
            img_seq=np.concatenate((img_seq,img),axis=0)
        #extract annotation files
        handJoints3D=np.empty((0,21,3))
        for pos in range(start_pos,start_pos+self.seq_len):
            pkl=utils.read_annotation(self.set_path,dir_name,f"{pos:0{4}}")
            hj=pkl['handJoints3D']
            hj=np.expand_dims(hj,axis=0)
            handJoints3D=np.concatenate((handJoints3D,hj),axis=0)
        print('pkl')






    

        return 1
    
d=Ho3D()
d[3]
    






