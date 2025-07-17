import numpy as np

class ReplayBuffer:
    def __init__(self, capacidade_maxima, obs_dim, acao_dim):
        self.capacidade_maxima = capacidade_maxima
        self.ptr = 0
        self.size = 0
        self.estados = np.zeros((capacidade_maxima, obs_dim), dtype=np.float32)
        self.acoes = np.zeros((capacidade_maxima, acao_dim), dtype=np.float32)
        self.recompensas = np.zeros(capacidade_maxima, dtype=np.float32)
        self.proximos_estados = np.zeros((capacidade_maxima, obs_dim), dtype=np.float32)
        self.finalizado = np.zeros(capacidade_maxima, dtype=np.bool_)

    def adicionar(self, estado, acao, recompensa, proximo_estado, finalizado):
        self.estados[self.ptr] = estado
        self.acoes[self.ptr] = acao
        self.recompensas[self.ptr] = recompensa
        self.proximos_estados[self.ptr] = proximo_estado
        self.finalizado[self.ptr] = finalizado
        self.ptr = (self.ptr + 1) % self.capacidade_maxima
        self.size = min(self.size + 1, self.capacidade_maxima)

    def amostrar(self, batch_size):
        indices_aleatorios = np.random.randint(0, self.size, size=batch_size)
        return (
            self.estados[indices_aleatorios],
            self.acoes[indices_aleatorios],
            self.recompensas[indices_aleatorios],
            self.proximos_estados[indices_aleatorios],
            self.finalizado[indices_aleatorios]
        )

    def __len__(self):
        return self.size
