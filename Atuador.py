import torch
import torch.nn as nn
import torch.nn.functional as F

#Classe responsável por criar novas ações (7-upla)

class Atuador (nn.Module):
    #obs_dim: dimensão do espaço de observação.
    #acao_dim: dimensão do espaço de ações.
    #acao_limite: limite númerico que uma ação pode ter. 
    #Ex.: [-acao_limite, acao_limite]
    def __init__(self, obs_dim, acao_dim, acao_limite):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, acao_dim),
            nn.Tanh()  # Para limitar entre [-1, 1]
        )

        self.acao_limite = acao_limite

    def forward(self, obs):
        # obs: tensor de shape [batch_size, obs_dim]
        # saída: ação contínua entre [-acao_limite, acao_limite]
        return self.net(obs) * self.acao_limite