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
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Inicialização leve para a última camada
        self.net[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.net[-1].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, acao):
        entrada = torch.cat([obs, acao], dim=-1)
        q_valor = self.net(entrada)
        return q_valor
