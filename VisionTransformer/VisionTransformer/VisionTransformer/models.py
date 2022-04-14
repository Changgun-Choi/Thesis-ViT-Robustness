import torch
import torch.nn as nn
torch.randn(1, 10)


"1. Make Input: patchembedding"
class LinearProjection(nn.Module):  

    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim)    # latent vector
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim))   # 학습가능 parameter, size(1, D)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim)) # num_patches+1: size(1, )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)  # (b x n x p2c) 
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1)  
        x += self.pos_embedding  
        x = self.dropout(x)
        return x    # b x n x d
#%%
"2. Multiheaded: 다른 위치에 대한 attention 능력 향상, num_heads 만큼 나누어서 병렬적으로 Attention 진행하고 Concat해서 이어줌: 다양한 특징에 대한 어텐션을 볼 수 있게 한 방법"
# scaled dot product attention 연산을 해주고 다시 원래의 행렬 모양으로 돌려주기 위해 axis=0으로 분리 후, axis=-1로 concatenate 해줌 

class MultiheadedSelfAttention(nn.Module):                  
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim 
        self.head_dim = int(latent_vec_dim / num_heads)  
              
        # 중요 포인트!! 모든 head 의 q,k,v를 구한것 
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)  # 원래는 self.head_dim 이 와서 각각인데 한번에 계산 
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)      
        self.scale = torch.sqrt(self.head_dim*torch.ones(1)).to(device)  # gpu 텐서
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
                                   # latent_vec_dim 을 self.num_heads, self.head_dim 로 나눔!!
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) # k 
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.t (k transpose)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) # q랑 동일
        attention = torch.softmax(q @ k / self.scale, dim=-1)  #  @는 Matrix 곱
        
        x = self.dropout(attention) @ v 
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim)

        return x, attention # attention:  저장용
#%%
"3. 여러개의 Transformer Layer 이어줌"
# Transformer Layer 1  
class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))
    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z   
        z = self.ln2(x)
        z = self.mlp(z)   # 100 image x 10 class
        x = x + z
        
        return x, att

#%% Main 
class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super().__init__()
        self.patchembedding = LinearProjection(patch_vec_size=patch_vec_size, num_patches=num_patches,     # Class token, patch embedding
                                               latent_vec_dim=latent_vec_dim, drop_rate=drop_rate)
        # 여러개 Layer 쌓기: Iterative 반복 
        self.transformer = nn.ModuleList([TFencoderLayer(latent_vec_dim=latent_vec_dim, num_heads=num_heads,
                                                         mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                          for _ in range(num_layers)])
                            #[ For문]으로 12번 TFencoderLayer 반복해서 list만듬
                                        
        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), nn.Linear(latent_vec_dim, num_classes))  # Class node에 맞춰서 output 

    def forward(self, x):
        att_list = []
        x = self.patchembedding(x)     # LinearProjection: Input
        for layer in self.transformer: # nn.ModuleList에서 List로 쌓여있는 것을 for문으로 불러옴
            x, att = layer(x)
            att_list.append(att)       # Attention weight for each layer
        x = self.mlp_head(x[:,0])      # x[:,0] : Class Token을 의미

        return x, att_list             # X 는 class
