import torch as torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1,p = 0.1):
        super(Model, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 1st encoder in a siamese fashion
        encoder = []
        for _ in range(num_layers):
            encoder.append(nn.Linear(input_dim, input_dim))
            encoder.append(nn.SELU())
            encoder.append(nn.AlphaDropout(p=p))

        # Final linear layer
        encoder.append(nn.Linear(input_dim, output_dim)) 

        # Create encoder
        self.encoder = nn.Sequential(*encoder)
        # output dropout
        
        # 2nd transformed by layernorm
        self.ln =  nn.LayerNorm(output_dim,elementwise_affine=False)
        
        # 3rd associated via similarity function (dot,cosine,...)
        self.cosine = nn.CosineSimilarity(dim = 0) #(dim=1, eps=1e-08)
        # dot used in forward
    
    def forward(self, query_mol, p_supp, n_supp,train=True):

        # Step 1 & 2: 
        p = self.ln(self.encoder(p_supp))
        n = self.ln(self.encoder(n_supp))
        q = self.ln(self.encoder(query_mol))

        # match the number of dimensions of p_supp and n_supp
        q = q.unsqueeze(1)  

        # Step 3: batch-wise dot product 
        q_p_dots = torch.sum(q * p, dim=-1)  
        q_n_dots = torch.sum(q * n, dim=-1)  

        q_p = torch.mean(q_p_dots,1)
        q_n = torch.mean(q_n_dots,1)

        # Step 4: skaling
        q_p *= 1 / self.input_dim ** 0.5
        q_n *= 1 / self.input_dim ** 0.5

        # step 5:
        prediction = torch.sigmoid(q_p-q_n)
        
        return prediction 