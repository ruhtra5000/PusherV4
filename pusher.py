import gymnasium as gym
import time
import torch
import numpy as np

from Atuador import Atuador

#Arquivo inicial

#Criação do ambiente PusherV4 e outras variáveis 
ambiente = gym.make("Pusher-v4", render_mode="human")

obsDim = ambiente.observation_space.shape[0] #Armazena a dimensão do espaço de observação (23)
acaoDim = ambiente.action_space.shape[0] #Armazena a dimensão do espaço de ações (7)
acaoLimite = 2 #Armazena o limite máximo de cada ação (intervalo [-2, 2])

estadoAtual = ambiente.reset()[0]  #Retorna o estado atual do ambiente

sigmaRuido = 0.2 #Parâmetro de ruído (desvio padrão da gaussiana)
decaimentoRuido = 0.98 #Diminui a qtde de ruido (conforme a exploração avança)

qtdeEpisodios = 50 
passosPorEpisodio = 100

#Criação do atuador
atuador = Atuador(obsDim, acaoDim, acaoLimite)

with torch.no_grad():
    for episodio in range(qtdeEpisodios):
        estadoAtual = ambiente.reset()[0]
        print(f'Episódio {episodio+1:2} | Ruído: {sigmaRuido:.4f}')

        #Episódio (100 time steps)
        for i in range(passosPorEpisodio):
            #Cria uma nova ação com o Atuador (passando um tensor torch como argumento)
            estadoTensor = torch.tensor(estadoAtual, dtype=torch.float32).unsqueeze(0)
            acao = atuador(estadoTensor).squeeze(0).numpy()

            #Adiciona ruído gaussiano (que garante a exploração)
            ruido = np.random.normal(0, sigmaRuido, size=acao.shape)
            acaoComRuido = acao + ruido

            #Executa a ação, retornando diversos dados, como:
            #O novo estado do ambiente, e a recompensa dada pela ação
            novoEstado, recompensa, finalizado, truncado, info = ambiente.step(acaoComRuido)

            if finalizado or truncado:
                break

            estadoAtual = novoEstado
            time.sleep(0.01)
        
        #Aplica o decaimento no ruído
        sigmaRuido *= decaimentoRuido

ambiente.close()