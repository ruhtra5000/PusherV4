import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

class Critico(nn.Module):

    #obs_dim: dimensão do espaço de observação.
    #acao_dim: dimensão do espaço de ações.
    def __init__(self, obs_dim, acao_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + acao_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, acao):
        entrada = torch.cat([obs, acao], dim=-1)
        q_valor = self.net(entrada)
        return q_valor
