import gymnasium as gym
from stable_baselines3 import SAC

# Cria o ambiente Pusher-v4
env = gym.make("Pusher-v4", render_mode="human")
# Cria o modelo SAC
model = SAC("MlpPolicy", env, verbose=1)
# Treinamento
model.learn(total_timesteps = 250_000)
# Salva o modelo
model.save("sac_pusher_v4")

# Teste opcional
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()

# Fecha o ambiente
env.close()
