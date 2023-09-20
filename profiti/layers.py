import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

def dense_layers(n_layers: int = 1, latent_dim: int = 2, device: str = 'cuda') -> nn.Sequential:
    layers = []
    for i in range(n_layers - 1):
        layers.append(nn.Linear(latent_dim, latent_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(latent_dim, 1))
    return nn.Sequential(*layers).to(device)

class Q_MAB(nn.Module):
    def __init__(self, q_in_dims, k_in_dims, latent_dims, nheads=2):
        super(Q_MAB, self).__init__()
        self.num_heads = nheads
        self.q_in_dims = q_in_dims
        self.k_in_dims = k_in_dims
        self.latent_dims = latent_dims
        self.d_k = latent_dims // nheads
        self.relu = nn.ReLU()
        self.query_linear = nn.Linear(q_in_dims, latent_dims)
        self.key_linear = nn.Linear(k_in_dims, latent_dims)
        self.value_linear = nn.Linear(k_in_dims, latent_dims)
        self.output_linear = nn.Linear(latent_dims, latent_dims)   

    def forward(self, query, key, mask):
        batch_size = query.size(0)
        query = self.relu(self.query_linear(query)) # Linera projection of queries, keys and values
        value = self.relu(self.value_linear(key))
        key = self.relu(self.key_linear(key))
        
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)         # Reshape query, key, and value to separate heads
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)) # Calculate attention scores

        mask = mask[:,None,:,:].repeat(1,scores.shape[1],1,1) # Masking the scores
        scores = scores.masked_fill(mask == 0, -1e8)

        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.output_linear(attended_values)
        return output

class grafiti(nn.Module):
    def __init__(self, dim, latent_dim, n_layers, f_layers, attn_head, device='cuda'):
        super(grafiti, self).__init__()
        self.dim = dim+2
        self.nheads = attn_head
        self.latent_dim = latent_dim
        self.edge_init = nn.Linear(2, latent_dim)
        self.chan_init = nn.Linear(dim, latent_dim)
        self.time_init = nn.Linear(1, latent_dim)
        self.n_layers = n_layers
        self.f_layers = f_layers
        self.channel_time_attn = nn.ModuleList()
        self.edge_nn = nn.ModuleList()
        self.device = device
        self.output = nn.Linear(3*latent_dim, latent_dim)
        for i in range(self.n_layers):
            self.channel_time_attn.append(Q_MAB(latent_dim, 2*latent_dim, latent_dim, self.nheads))
            self.edge_nn.append(nn.Linear(3*latent_dim, latent_dim))
        self.out_layer = nn.Linear(latent_dim, self.f_layers)
        self.relu = nn.ReLU()
    
    def gather(self, x, inds):
        # inds =  # keep repeating until the embedding len as a new dim
        return x.gather(1, inds[:,:,None].repeat(1,1,x.shape[-1]))

    def gatherhedge(self, U_, indices, mk_, shapes):
        # pdb.set_trace()
        X = torch.zeros([shapes[0],shapes[1],shapes[2], U_.shape[-1]]).to(U_.device)
        values = U_[mk_.to(torch.bool)]
        X[indices] = values
        return X

    def forward(self, TX: Tensor, X: Tensor, MX: Tensor, MY: Tensor):
        ''' TX: Tensor of shape BxT
            X: Tensor of shape BxTxC
            MX: Boolearn Tensor of shape BxTxC
            TY: Tensor of shape BxT
            MY: Tensor of shape BxTxC
        '''
        mask = MX + MY
        indices = torch.where(mask)
        ndims = X.shape[-1] # C
        T = TX[:,:,None] # BxTx1
        C = torch.ones([TX.shape[0], ndims]).cumsum(1).to(self.device) - 1 #BxC intialization for one hot encoding channels
        T_inds = torch.cumsum(torch.ones_like(X).to(torch.int64), 1) - 1 #BxTxC init for time indices
        C_inds = torch.cumsum(torch.ones_like(X).to(torch.int64), -1) - 1 #BxTxC init for channel indices
        mk_bool = mask.to(torch.bool) # BxTxC
        full_len = torch.max(mask.sum((1,2))).to(torch.int64) # flattened TxC max length possible
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0)

        # flattening to 2D
        T_inds_ = torch.stack([pad(r[m]) for r, m in zip(T_inds, mk_bool)]).contiguous() #BxTxC -> Bxfull_len 
        U_ = torch.stack([pad(r[m]) for r, m in zip(X, mk_bool)]).contiguous() #BxTxC (values) -> Bxfull_len 
        target_mask_ = torch.stack([pad(r[m]) for r, m in zip(MY, mk_bool)]).contiguous() #BxK_
        C_inds_ = torch.stack([pad(r[m]) for r, m in zip(C_inds, mk_bool)]).contiguous() #BxK_
        mk_ = torch.stack([pad(r[m]) for r, m in zip(mask, mk_bool)]).contiguous() #BxK_

        obs_len = full_len

        C_ = torch.nn.functional.one_hot(C.to(torch.int64), num_classes=ndims).to(torch.float32) #BxCxC #channel one hot encoding
        U_indicator = 1-mk_+target_mask_
        U_ = torch.cat([U_[:,:,None], U_indicator[:,:,None]], -1) #BxK_max x 2 #todo: correct

        
        # creating Channel mask and Time mask
        C_mask = C[:,:,None].repeat(1,1,obs_len)
        temp_c_inds = C_inds_[:,None,:].repeat(1,ndims,1)
        C_mask = (C_mask == temp_c_inds).to(torch.float32) #BxCxK_
        C_mask = C_mask*mk_[:,None,:].repeat(1,C_mask.shape[1],1)

        T_mask = T_inds_[:,None,:].repeat(1,T.shape[1],1)
        temp_T_inds = torch.ones_like(T[:,:,0]).cumsum(1)[:,:,None].repeat(1,1,C_inds_.shape[1]) -1
        T_mask = (T_mask == temp_T_inds).to(torch.float32) #BxTxK_
        T_mask = T_mask*mk_[:,None,:].repeat(1,T_mask.shape[1],1)
        U_ = self.relu(self.edge_init(U_)) * mk_[:,:,None].repeat(1,1,self.latent_dim) # 
        T_ = torch.sin(self.time_init(T)) # learned time embedding
        C_ = self.relu(self.chan_init(C_)) # embedding on one-hot encoded channel

        del temp_T_inds
        del temp_c_inds
        C_mask_ = C_mask.sum(-1).bool().float()
        T_mask_ = T_mask.sum(-1).bool().float()
        
        # hedge = []
        for i in range(self.n_layers):

            # channels as queries
            q_c = C_
            k_t = self.gather(T_, T_inds_) # BxK_max x embd_len
            k = torch.cat([k_t, U_], -1) # BxK_max x 2 * embd_len

            C__ = self.channel_time_attn[i](q_c, k, C_mask) # attn (channel_embd, concat(time, values)) along with the mask
            C__ = C__*(C_mask_[:,:,None].repeat(1,1,C__.shape[-1]))
            # times as queries
            q_t = T_
            k_c = self.gather(C_, C_inds_)
            k = torch.cat([k_c, U_], -1)
            T__ = self.channel_time_attn[i](q_t, k, T_mask)
            T__ = T__*(T_mask_[:,:,None].repeat(1,1,T__.shape[-1]))
            # updating edge weights
            U_ = self.relu(U_ + self.edge_nn[i](torch.cat([U_, k_t, k_c], -1))) * mk_[:,:,None].repeat(1,1,self.latent_dim)
            C_ = C__
            T_ = T__
        U_rs = self.gatherhedge(U_, indices, mk_, mask.shape)
        return U_rs