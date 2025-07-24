import gymnasium as gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from Atuador import Atuador
from Critico import Critico
from ReplayBuffer import ReplayBuffer

#Arquivo inicial

#Criação do ambiente PusherV4 e outras variáveis 
ambiente = gym.make("Pusher-v4", render_mode=None) # "human"

obsDim = ambiente.observation_space.shape[0] #Armazena a dimensão do espaço de observação (23)
acaoDim = ambiente.action_space.shape[0] #Armazena a dimensão do espaço de ações (7)
acaoLimite = 2 #Armazena o limite máximo de cada ação (intervalo [-2, 2])

estadoAtual = ambiente.reset()[0]  #Retorna o estado atual do ambiente

sigmaRuido = 0.2 #Parâmetro de ruído (desvio padrão da gaussiana)
decaimentoRuido = 0.997 #Diminui a qtde de ruido (conforme a exploração avança)

qtdeEpisodios = 1000 
passosPorEpisodio = 100

# Parâmetros de Treinamento DDPG
gamma = 0.99  # Fator de desconto para recompensas futuras
tau = 0.003   # Taxa de atualização suave para as redes alvo
lr_atuador = 1e-4 # Taxa de aprendizado do atuador
lr_critico = 5e-4 # Taxa de aprendizado do crítico
batch_size = 256  # Tamanho do lote para amostragem do ReplayBuffer
delay_treinamento = 10000 # Número de passos antes de começar a treinar (para preencher o buffer)
atualizacoes_por_passo = 3 # Número de vezes que o agente treina por passo de ambiente

# Criação do atuador e crítico
atuador = Atuador(obsDim, acaoDim, acaoLimite)
critico = Critico(obsDim, acaoDim)

# Criação das redes alvo (target networks)
atuador_alvo = Atuador(obsDim, acaoDim, acaoLimite)
critico_alvo = Critico(obsDim, acaoDim)

# Inicializa as redes alvo com os mesmos pesos das redes principais
atuador_alvo.load_state_dict(atuador.state_dict())
critico_alvo.load_state_dict(critico.state_dict())

# Otimizadores
otimizador_atuador = optim.Adam(atuador.parameters(), lr=lr_atuador)
otimizador_critico = optim.Adam(critico.parameters(), lr=lr_critico)

# Instancia o ReplayBuffer 
buffer_capacidade = 200_000
buffer = ReplayBuffer(buffer_capacidade, obsDim, acaoDim)

total_passos = 0 # Contador para o número total de passos no ambiente

for episodio in range(qtdeEpisodios):
    estadoAtual = ambiente.reset()[0]
    print(f'Episódio {episodio+1:2} | Ruído: {sigmaRuido:.4f}')

    recompensa_acumulada = 0 # Para monitorar o progresso

    #Episódio (100 time steps)
    for i in range(passosPorEpisodio):
        total_passos += 1

        # Seleção de Ação (Exploração vs. Explotação)
        # O atuador não precisa de torch.no_grad() aqui porque o gradiente é necessário para o treino do atuador
        # Mas para a coleta de dados, a ação em si não é usada para treinar o atuador ainda.
        # No entanto, em DDPG, a ação do atuador é determinística e o ruído é adicionado explicitamente para exploração.
        estadoTensor = torch.tensor(estadoAtual, dtype=torch.float32).unsqueeze(0)
        
        # Gera a ação determinística do atuador
        with torch.no_grad(): # Não calcular gradientes para a coleta de dados
            acao_determinista = atuador(estadoTensor).squeeze(0).numpy()

        # Adiciona ruído gaussiano (que garante a exploração)
        ruido = np.random.normal(0, sigmaRuido, size=acao_determinista.shape)
        acaoComRuido = acao_determinista + ruido
        
        # Garante que a ação esteja dentro dos limites do ambiente
        acaoComRuido = np.clip(acaoComRuido, -acaoLimite, acaoLimite)

        # Executa a ação no ambiente
        novoEstado, recompensa, finalizado, truncado, info = ambiente.step(acaoComRuido)
        recompensa_acumulada += recompensa

        # Adiciona transição no ReplayBuffer
        buffer.adicionar(estadoAtual, acaoComRuido, recompensa, novoEstado, finalizado or truncado)

        # Lógica de Treinamento
        if total_passos >= delay_treinamento and len(buffer) >= batch_size:
            for _ in range(atualizacoes_por_passo):
                # 1. Amostrar um lote do buffer
                batch_estados_np, batch_acoes_np, batch_recompensas_np, \
                batch_proximos_estados_np, batch_finalizados_np = buffer.amostrar(batch_size)

                # 2. Converter arrays NumPy para tensores PyTorch
                batch_estados = torch.tensor(batch_estados_np, dtype=torch.float32)
                batch_acoes = torch.tensor(batch_acoes_np, dtype=torch.float32)
                batch_recompensas = torch.tensor(batch_recompensas_np, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]
                batch_proximos_estados = torch.tensor(batch_proximos_estados_np, dtype=torch.float32)
                batch_finalizados = torch.tensor(batch_finalizados_np, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]

                # 3. Treinar o Crítico
                otimizador_critico.zero_grad() # Zera os gradientes
                
                # Calcular o alvo Q (Y_t) usando as redes alvo para estabilidade
                with torch.no_grad(): # Não calcular gradientes para o alvo Q
                    proximas_acoes = atuador_alvo(batch_proximos_estados)
                    proximos_q_valores = critico_alvo(batch_proximos_estados, proximas_acoes)
                    
                    # Se o episódio terminou, o Q do próximo estado é 0
                    alvo_q = batch_recompensas + gamma * (1 - batch_finalizados) * proximos_q_valores
                    alvo_q = torch.clamp(alvo_q, min=-100.0, max=100.0)

                q_predito = critico(batch_estados, batch_acoes) # Q-valor predito pelo crítico principal
                
                if total_passos % 10 == 0:
                    print(f"Alvo Q: min {alvo_q.min().item():.4f}, max {alvo_q.max().item():.4f}, mean {alvo_q.mean().item():.4f}")
                    print(f"Q predito: min {q_predito.min().item():.4f}, max {q_predito.max().item():.4f}, mean {q_predito.mean().item():.4f}")

                perda_critico = F.mse_loss(q_predito, alvo_q) # Erro quadrático médio
                perda_critico.backward() # Backpropagation
                otimizador_critico.step() # Atualiza os pesos do crítico

                # 4. Treinar o Atuador
                otimizador_atuador.zero_grad() # Zera os gradientes

                # As ações são geradas pelo atuador principal para maximizar o Q predito pelo crítico principal
                acoes_atuador = atuador(batch_estados)
                perda_atuador = -critico(batch_estados, acoes_atuador).mean() # Maximiza o Q -> minimiza -Q
                
                perda_atuador.backward() # Backpropagation
                otimizador_atuador.step() # Atualiza os pesos do atuador
                
                if total_passos % 10 == 0:
                    print(f"[Ep {episodio+1:4} | Passo {total_passos:6}] Perda Crítico: {perda_critico.item():.4f} | Perda Atuador: {perda_atuador.item():.4f}")

                # 5. Atualizar as Redes Alvo (Soft Update)
                # target_weights = tau * main_weights + (1 - tau) * target_weights
                with torch.no_grad(): # As atualizações das redes alvo não precisam de gradientes
                    for param, target_param in zip(critico.parameters(), critico_alvo.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(atuador.parameters(), atuador_alvo.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


        if finalizado or truncado:
            break

        estadoAtual = novoEstado
        #time.sleep(0.0005) # Pequeno atraso para visualização
    
    # Aplica o decaimento no ruído
    sigmaRuido = max(sigmaRuido * decaimentoRuido, 0.05)
    
    print(f"  Recompensa Acumulada no Episódio: {recompensa_acumulada:.2f}")

ambiente.close()