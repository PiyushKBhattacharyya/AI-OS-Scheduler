import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import time

# Add project root to path for modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.scheduler_env import ProcessEnv
from src.schedulers.scheduler import FCFS, RR

def evaluate_rl(model_path, n_processes=5):
    env = ProcessEnv(max_queue_size=5, tick_duration=0.1)
    model = PPO.load(model_path)
    
    obs, _ = env.reset(options={'n_processes': n_processes})
    done = False
    
    print(f"Evaluating RL Agent on {n_processes} real processes...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    results = env.completed_processes
    env.close()
    return results

def compare_and_plot():
    n_proc = 5
    
    # 1. Run FCFS
    fcfs_sched = FCFS(n_processes=n_proc)
    fcfs_results = fcfs_sched.run()
    fcfs_wait = np.mean([p.completion_time - p.start_wall_time - p.burst_time for p in fcfs_results])
    
    # 2. Run RR
    rr_sched = RR(n_processes=n_proc, quantum=0.2)
    rr_results = rr_sched.run()
    rr_wait = np.mean([p.completion_time - p.start_wall_time - p.burst_time for p in rr_results])
    
    # 3. Run RL
    model_path = "ppo_scheduler"
    if os.path.exists(model_path + ".zip"):
        rl_results = evaluate_rl(model_path, n_processes=n_proc)
        rl_wait = np.mean([p.completion_time - p.start_wall_time - p.burst_time for p in rl_results])
    else:
        print("RL model not found. Run scripts/train.py first.")
        rl_wait = 0

    # Visualization
    labels = ['FCFS', 'Round Robin', 'RL Agent']
    waits = [fcfs_wait, rr_wait, rl_wait]
    
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, waits, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.ylabel('Average Wait Time (s)')
    plt.title('Real Process Performance Comparison')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')
    
    plt.savefig('results/performance_comparison.png')
    print("Performance plot saved in results/performance_comparison.png")
    
    # Wait time distribution
    plt.figure(figsize=(10, 6))
    plt.hist([p.completion_time - p.start_wall_time - p.burst_time for p in fcfs_results], alpha=0.5, label='FCFS')
    plt.hist([p.completion_time - p.start_wall_time - p.burst_time for p in rr_results], alpha=0.5, label='RR')
    if rl_wait > 0:
        plt.hist([p.completion_time - p.start_wall_time - p.burst_time for p in rl_results], alpha=0.5, label='RL')
    
    plt.xlabel('Wait Time (s)')
    plt.ylabel('Frequency')
    plt.title('Wait Time Distribution (Real Processes)')
    plt.legend()
    plt.savefig('results/wait_time_distribution.png')
    print("Wait time distribution saved in results/wait_time_distribution.png")

if __name__ == "__main__":
    compare_and_plot()
