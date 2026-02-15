import gymnasium as gym
from stable_baselines3 import PPO
from scheduler_env import SchedulerEnv
import os

def train():
    # Initialize environment
    env = SchedulerEnv(max_queue_size=10)
    
    # Define and train the model
    # Using PPO (Proximal Policy Optimization) - good for discovery in discrete spaces
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Starting training...")
    model.learn(total_timesteps=20000)
    
    # Save the model
    model.save("ppo_scheduler")
    print("Model saved as ppo_scheduler.zip")

if __name__ == "__main__":
    train()
