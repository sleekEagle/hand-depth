import torch
import data.FreiHAND.dataset as dataset
from torch.utils.data import DataLoader
from models import model
import torch.nn as nn
from omegaconf import DictConfig

def indexify_joints(joints):
    idx_joints={}
    n=0
    for i,key_1 in enumerate(list(joints.keys())):
        if type(joints[key_1])==DictConfig:
            for j,sub_key in enumerate(list(joints[key_1].keys())):
                idx_joints[n]=f'{key_1}-{sub_key}'
                n+=1
        elif type(joints[key_1])==int:
            idx_joints[n]=key_1
            n+=1
    return idx_joints

class Trainer():
    def __init__(self,conf):
        self.conf=conf
        self.loss_func=nn.MSELoss()
        if conf.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device='cpu'
        print(f'using device {self.device}')


    def get_optimizer(self,model):
        model_params=[p for n, p in model.named_parameters() if p.requires_grad]
       
        optimizer = torch.optim.AdamW(model_params, lr=self.conf.train.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=self.conf.train.lr_decay_step,
                                                       gamma=self.conf.train.lr_dec_factor)
        return optimizer, lr_scheduler
    
    def _make_model(self):
        self.model=model.Model(self.conf,self.device)
        self.model.to(self.device)
        self.optimizer, self.lr_scheduler = self.get_optimizer(self.model)
        
    def _make_dataloader(self):
        d_tr=dataset.FreiHAND(self.conf,mode='training')
        train_dataloader = DataLoader(d_tr, batch_size=self.conf.datasets.freihand.bs, 
                                        num_workers=self.conf.train.num_thread,
                                        shuffle=True)
        self.data_loader=train_dataloader
    
    def train_epoch(self):
        for itr,inputs in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            model_out=self.model(inputs)
            loss=self.loss_func(model_out.double(),(inputs['dists'].double()).to(self.device))
            loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()
       

class Tester():
    def __init__(self,conf):
        self.conf=conf
        if conf.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device='cpu'
    
    def _make_dataloader(self):
        d_eval=dataset.FreiHAND(self.conf,mode='evaluation')
        eval_dataloader = DataLoader(d_eval, batch_size=1, 
                                        num_workers=1,
                                        shuffle=True)
        self.data_loader=eval_dataloader

    def set_model(self,model):
        self.model=model
    
    def evaluate(self):
        errors=torch.zeros((1,self.conf.datasets.freihand.num_joints)).to(self.device)
        for itr,inputs in enumerate(self.data_loader):
            model_out=self.model(inputs)
            gt_depth=inputs['dists'].to(self.device)

            if self.conf.eval.eval_idx==-1:
                selected_model_out=model_out
                selected_gt_depth=gt_depth
            else:
                selected_model_out=model_out[self.conf.eval.eval_idx].unsqueeze(dim=0)
                selected_gt_depth=gt_depth[self.conf.eval.eval_idx].unsqueeze(dim=0)

            rmse=torch.sqrt(torch.mean(torch.square(selected_model_out-selected_gt_depth),dim=0)).unsqueeze(0)
            errors+=rmse
        joint_rmse_error=errors/len(self.data_loader)
        mean_rmse_error=torch.mean(errors).item()

        #indexify the joint dict
        joints=self.conf.datasets.freihand.joint_idx
        idx_joints=indexify_joints(joints)

        #print the joint-wise errors
        print('Joint-wise RMSE in meters:')
        for i in range(joint_rmse_error.shape[1]):
            print(f"{idx_joints[i]}: %.3f" % joint_rmse_error[0,i].item())

        print(f'mean RMSE in meters :{mean_rmse_error}')

        return mean_rmse_error,joint_rmse_error



    


