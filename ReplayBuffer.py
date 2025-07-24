import numpy as np
import os

class ReplayBuffer:
    """
    Buffer de repetição para armazenar transições (estado, ação, recompensa, próximo estado, finalização)
    usado em algoritmos de aprendizado por reforço, como DDPG ou TD3.
    """

    def __init__(self, capacidade_maxima, obs_dim, acao_dim, seed=None):
        self.capacidade_maxima = capacidade_maxima
        self.ptr = 0
        self.size = 0
        self.obs_dim = obs_dim
        self.acao_dim = acao_dim
        self.rng = np.random.default_rng(seed)

        # Dados armazenados
        self.estados = np.zeros((capacidade_maxima, obs_dim), dtype=np.float32)
        self.acoes = np.zeros((capacidade_maxima, acao_dim), dtype=np.float32)
        self.recompensas = np.zeros(capacidade_maxima, dtype=np.float32)
        self.proximos_estados = np.zeros((capacidade_maxima, obs_dim), dtype=np.float32)
        self.finalizado = np.zeros(capacidade_maxima, dtype=bool)

        # Estatísticas para normalização
        self.soma_estados = np.zeros(obs_dim, dtype=np.float64)
        self.soma_quadrado_estados = np.zeros(obs_dim, dtype=np.float64)

    def adicionar(self, estado, acao, recompensa, proximo_estado, finalizado):
        """Adiciona uma transição ao buffer."""
        self.estados[self.ptr] = estado
        self.acoes[self.ptr] = acao
        self.recompensas[self.ptr] = recompensa
        self.proximos_estados[self.ptr] = proximo_estado
        self.finalizado[self.ptr] = finalizado

        # Atualização incremental de estatísticas
        self.soma_estados += estado
        self.soma_quadrado_estados += estado ** 2

        # Atualização de ponteiro e tamanho
        self.ptr = (self.ptr + 1) % self.capacidade_maxima
        self.size = min(self.size + 1, self.capacidade_maxima)

    def amostrar(self, batch_size):
        """Amostra um mini-batch aleatório do buffer."""
        if self.size < batch_size:
            raise ValueError(f"Buffer contém apenas {self.size} amostras, necessário {batch_size}.")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.estados[indices],
            self.acoes[indices],
            self.recompensas[indices],
            self.proximos_estados[indices],
            self.finalizado[indices]
        )

    def media_estado(self):
        """Retorna a média dos estados armazenados (para normalização)."""
        if self.size == 0:
            return np.zeros(self.obs_dim, dtype=np.float32)
        media = self.soma_estados / self.size
        return (media).astype(np.float32)

    def desvio_estado(self):
        """Retorna o desvio padrão dos estados armazenados (para normalização)."""
        if self.size == 0:
            return np.ones(self.obs_dim, dtype=np.float32)
        media = self.soma_estados / self.size
        media_quadrado = media ** 2
        media_quadrado_estado = self.soma_quadrado_estados / self.size
    
        variancia = media_quadrado_estado - media_quadrado
        variancia = np.maximum(variancia, 1e-6)  # Previne raiz de número negativo ou zero
        desvio = np.sqrt(variancia)

        return desvio.astype(np.float32)

    def salvar(self, caminho):
        """Salva o buffer em um arquivo .npz comprimido."""
        np.savez_compressed(caminho,
                            estados=self.estados[:self.size],
                            acoes=self.acoes[:self.size],
                            recompensas=self.recompensas[:self.size],
                            proximos_estados=self.proximos_estados[:self.size],
                            finalizado=self.finalizado[:self.size],
                            soma_estados=self.soma_estados,
                            soma_quadrado_estados=self.soma_quadrado_estados,
                            size=self.size,
                            ptr=self.ptr,
                            obs_dim=self.obs_dim,
                            acao_dim=self.acao_dim)
        print(f"ReplayBuffer salvo em: {caminho}")

    def carregar(self, caminho):
        """Carrega o buffer a partir de um arquivo .npz."""
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo '{caminho}' não encontrado.")

        data = np.load(caminho)

        self.size = int(data["size"])
        self.ptr = int(data["ptr"])

        # Verificação de dimensões
        assert data["estados"].shape[1] == self.obs_dim, "Dimensão de estado incompatível."
        assert data["acoes"].shape[1] == self.acao_dim, "Dimensão de ação incompatível."

        self.estados[:self.size] = data["estados"]
        self.acoes[:self.size] = data["acoes"]
        self.recompensas[:self.size] = data["recompensas"]
        self.proximos_estados[:self.size] = data["proximos_estados"]
        self.finalizado[:self.size] = data["finalizado"]

        self.soma_estados = data["soma_estados"]
        self.soma_quadrado_estados = data["soma_quadrado_estados"]

        print(f"ReplayBuffer carregado de: {caminho}")

    def __len__(self):
        """Retorna o número atual de elementos no buffer."""
        return self.size
