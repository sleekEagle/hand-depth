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
    # rmse_error=tester.evaluate()

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

# trainer=Trainer(c)
# trainer._make_dataloader()

# d=iter(trainer.data_loader).next()
# import torch
# kypts=torch.permute(d['xyz'],(2,1,0))

# hand_dims=utils.get_hand_dims(kypts,c)




































