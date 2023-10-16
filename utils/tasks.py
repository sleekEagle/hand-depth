import torch
import data.FreiHAND.dataset as dataset
from torch.utils.data import DataLoader
from models import model
import torch.nn as nn

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
        errors=torch.tensor(0.0).to(self.device)
        for itr,inputs in enumerate(self.data_loader):
            model_out=self.model(inputs)
            gt_depth=inputs['dists'].to(self.device)
            rmse=torch.sqrt(torch.mean(torch.square(model_out-gt_depth)))
            errors+=rmse
        rmse_error=errors/len(self.data_loader)
        print(f'RMSE in m :{rmse_error.item()}')
        return rmse_error.item()



    


