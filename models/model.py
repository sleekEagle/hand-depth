import torch.nn.functional as F
from torch import nn, Tensor
import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import utils.utils

#used to get position embedding of locations of a 2D array (eg. an image)
#copied from keypoint transformer https://github.com/shreyashampali/kypt_transformer
#img is just used to get its device. 

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=100, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self,mask):
        assert mask is not None
        not_mask = ~(mask.squeeze(1)>0)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # N x hidden_dim x H x W

        return pos

class Model(nn.Module):
    def __init__(self,conf,device):
        super(Model, self).__init__()
        self.conf=conf
        self.device=device
        dataset_conf=conf.datasets[conf.dataset]
        self.num_joints=dataset_conf.num_joints

        #obtain positional encoding
        self.position_embedding = PositionEmbeddingSine(dataset_conf.model.encoder.pos_embed_dim, normalize=True)
        mask=torch.zeros((1,1,dataset_conf.model.pos_mask_size,dataset_conf.model.pos_mask_size))
        self.pos=self.position_embedding(mask)
        self.joint_embed = nn.Embedding(self.num_joints, dataset_conf.model.encoder.joint_embed_dim)

        d_model=dataset_conf.model.encoder.pos_embed_dim*2+dataset_conf.model.encoder.joint_embed_dim
        if conf.train.hand_dim.use_hand_dim:
            if not conf.train.hand_dim.late_fusion:
                d_model=conf.train.hand_dim.d_model
                #personal hand dimention is 20-D (distance to various joints from parent joint. there are 21 joints.)
                self.early_linear = nn.Linear(self.num_joints-1 +
                                            dataset_conf.model.encoder.pos_embed_dim*2 +
                                            dataset_conf.model.encoder.joint_embed_dim ,
                                            d_model)
            else:
                #late fusion
                self.late_fusion_linear=nn.Linear(d_model+self.num_joints-1, 1)

        
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                nhead=dataset_conf.model.encoder.nhead, 
                                                dim_feedforward=dataset_conf.model.encoder.dim_feedforward, 
                                                dropout=dataset_conf.model.encoder.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=dataset_conf.model.encoder.num_layers)
        self.linear = nn.Linear(d_model, 1)

    def get_pos_embedding(self,inputs):
        if self.conf.datasets[self.conf.dataset].gt_kpt_train:
            kypts=inputs['uv']
        else:
            kypts=inputs['unet_annot']

        #get root-relative 2D focal-distance normalized coordinates
        roots=torch.repeat_interleave(torch.unsqueeze(kypts[:,0,:],dim=1),repeats=kypts.shape[1],dim=1)
        fx=inputs['K'][:,0,0]
        fy=inputs['K'][:,1,1]
        fx=torch.unsqueeze(fx,dim=1).unsqueeze(dim=2)
        fx=torch.repeat_interleave(fx,repeats=kypts.shape[1],dim=1)
        fy=torch.unsqueeze(fy,dim=1).unsqueeze(dim=2)
        fy=torch.repeat_interleave(fy,repeats=kypts.shape[1],dim=1)
        f=torch.concat((fx,fy),dim=-1)
        rel_coord=(kypts-roots)/f
        grids=torch.unsqueeze(rel_coord,dim=1).float()
        bs=kypts.shape[0]
        pos=torch.repeat_interleave(self.pos,repeats=bs,dim=0)
        positions=nn.functional.grid_sample(pos,grids,mode='nearest', align_corners=True).squeeze(2)
        positions=positions.permute(0,2,1)
        return positions


    def forward(self,inputs):
        pos_embeddings=self.get_pos_embedding(inputs).to(self.device)
        bs=pos_embeddings.shape[0]
        joint_embeddings=self.joint_embed.weight
        joint_embeddings=torch.unsqueeze(joint_embeddings,dim=0)
        joint_embeddings=torch.repeat_interleave(joint_embeddings,repeats=bs,dim=0)
        features=torch.cat([pos_embeddings,joint_embeddings],dim=-1)
        if self.conf.train.hand_dim.use_hand_dim:
            hand_dims=inputs['hand_dim'].to(torch.float32)
            hand_dims=torch.unsqueeze(hand_dims,dim=1).repeat_interleave(repeats=self.num_joints,dim=1)
            hand_dims=hand_dims.to(self.device)
            if not self.conf.train.hand_dim.late_fusion:
                # do early fusion of personal hand embeddings
                features=torch.cat([features,hand_dims],dim=-1)                
                features=self.early_linear(features)
        encoder_out=self.transformer_encoder(features)
        if self.conf.train.hand_dim.use_hand_dim:
            if self.conf.train.hand_dim.late_fusion:
                enc_out_hand=torch.cat((encoder_out,hand_dims),dim=-1)
                pred=self.late_fusion_linear(enc_out_hand)
        pred=self.linear(encoder_out)
        pred=pred.squeeze(dim=-1)
        return pred





