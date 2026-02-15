import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO

# Add project root to path for modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.scheduler_env import ProcessEnv

def train():
    print("Initializing Real Process Environment for training...")
    # Initialize environment
    env = ProcessEnv(max_queue_size=5, tick_duration=0.1)
    
    # Define and train the model
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Starting training on real processes (this may take a few minutes)...")
    # In real-time, 1000 steps = 100 seconds of execution
    model.learn(total_timesteps=1000)
    
    # Save the model
    model.save("ppo_scheduler")
    print("Model saved as ppo_scheduler.zip")
    env.close()

if __name__ == "__main__":
    train()
