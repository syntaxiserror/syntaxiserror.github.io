import gym
import stable_baselines3
from stable_baselines3 import A2C

env = gym.make("LunarLander-v2", render_mode="human")
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

for _ in range(10):
    obs = env.reset()
    env.render()
    done = False

    while not done:
        obs, reward, done, info, s = env.step(env.action_space.sample())
env.close()
