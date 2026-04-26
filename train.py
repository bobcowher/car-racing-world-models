from agent import Agent
import gymnasium as gym

env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

agent = Agent(env=env, max_buffer_size=100000, target_update_interval=1000, dyna_k=5)

agent.train(episodes=1200, wm_batch_size=32, dqn_batch_size=64, wm_updates=5, warmup=25)
