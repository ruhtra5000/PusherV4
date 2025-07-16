import gymnasium as gym
import time

ambiente = gym.make("Pusher-v4", render_mode="human")

obs, _ = ambiente.reset()

for _ in range(100):
    acao = ambiente.action_space.sample()
    print(f'ação {_}: {acao}')
    obs, recompensa, terminado, truncado, info = ambiente.step(acao)
    
    if terminado or truncado:
        obs, _ = ambiente.reset()
    
    time.sleep(0.01)

ambiente.close()