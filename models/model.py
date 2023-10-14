import torch.nn.functional as F
from torch import nn, Tensor
import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
    def __init__(self,conf):
        super(Model, self).__init__()
        self.conf=conf
        dataset_conf=conf.datasets[conf.dataset]

        #obtain positional encoding
        self.position_embedding = PositionEmbeddingSine(dataset_conf.model.encoder.pos_embed_dim, normalize=True)
        mask=torch.zeros((1,1,dataset_conf.model.pos_mask_size,dataset_conf.model.pos_mask_size))
        self.pos=self.position_embedding(mask)
        self.joint_embed = nn.Embedding(dataset_conf.num_joints, dataset_conf.model.encoder.joint_embed_dim)
        d_model=dataset_conf.model.encoder.pos_embed_dim*2 + dataset_conf.model.encoder.joint_embed_dim
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                  nhead=dataset_conf.model.encoder.nhead, 
                                                  dim_feedforward=dataset_conf.model.encoder.dim_feedforward, 
                                                  dropout=dataset_conf.model.encoder.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=dataset_conf.model.encoder.num_layers)
        self.linear = nn.Linear(d_model, 1)

    def get_pos_embedding(self,inputs):
        #get root-relative 2D focal-distance normalized coordinates
        roots=torch.repeat_interleave(torch.unsqueeze(inputs['uv'][:,0,:],dim=1),repeats=inputs['uv'].shape[1],dim=1)
        fx=inputs['K'][:,0,0]
        fy=inputs['K'][:,1,1]
        fx=torch.unsqueeze(fx,dim=1).unsqueeze(dim=2)
        fx=torch.repeat_interleave(fx,repeats=inputs['uv'].shape[1],dim=1)
        fy=torch.unsqueeze(fy,dim=1).unsqueeze(dim=2)
        fy=torch.repeat_interleave(fy,repeats=inputs['uv'].shape[1],dim=1)
        f=torch.concat((fx,fy),dim=-1)
        rel_coord=(inputs['uv']-roots)/f
        grids=torch.unsqueeze(rel_coord,dim=1).float()
        bs=inputs['uv'].shape[0]
        pos=torch.repeat_interleave(self.pos,repeats=bs,dim=0)
        positions=nn.functional.grid_sample(pos,grids,mode='nearest', align_corners=True).squeeze(2)
        positions=positions.permute(0,2,1)
        return positions


    def forward(self,inputs):
        pos_embeddings=self.get_pos_embedding(inputs)
        joint_embeddings=self.joint_embed.weight
        joint_embeddings=torch.unsqueeze(joint_embeddings,dim=0)
        joint_embeddings=torch.repeat_interleave(joint_embeddings,repeats=pos_embeddings.shape[0],dim=0)
        inputs=torch.cat([pos_embeddings,joint_embeddings],dim=-1)
        encoder_out=self.transformer_encoder(inputs)
        pred=self.linear(encoder_out)
        pred=pred.squeeze(dim=-1)
        return pred




