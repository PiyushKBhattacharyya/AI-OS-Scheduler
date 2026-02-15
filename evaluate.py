import gymnasium as gym
from stable_baselines3 import PPO
from scheduler_env import SchedulerEnv
from simulator import Process, FCFSSimulator, RRSimulator
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_test_processes(n=50):
    processes = []
    for i in range(n):
        arrival = np.random.uniform(0, 100)
        burst = np.random.uniform(1, 15)
        processes.append(Process(pid=i, arrival_time=arrival, burst_time=burst))
    return processes

def evaluate_rl(model_path, processes):
    env = SchedulerEnv(max_queue_size=10)
    model = PPO.load(model_path)
    
    obs, _ = env.reset(options={'processes': processes})
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    return env.completed_processes

def compare_and_plot():
    test_processes = generate_test_processes(30)
    
    # Run FCFS
    fcfs = FCFSSimulator(test_processes)
    fcfs_results = fcfs.run()
    fcfs_wait = np.mean([p.wait_time for p in fcfs_results])
    
    # Run Round Robin
    rr = RRSimulator(test_processes, quantum=2.0)
    rr_results = rr.run()
    rr_wait = np.mean([p.wait_time for p in rr_results])
    
    # Run RL (Assuming model exists)
    try:
        if not os.path.exists("ppo_scheduler.zip"):
            print("RL model not found. Run train.py first.")
            rl_wait = 0
        else:
            rl_results = evaluate_rl("ppo_scheduler", test_processes)
            rl_wait = np.mean([p.wait_time for p in rl_results])
    except Exception as e:
        print(f"Error evaluating RL: {e}")
        rl_wait = 0

    # Visualization
    labels = ['FCFS', 'Round Robin', 'RL Agent']
    waits = [fcfs_wait, rr_wait, rl_wait]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, waits, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.ylabel('Average Wait Time')
    plt.title('Performance Comparison: Traditional vs RL Scheduler')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
    
    plt.savefig('performance_comparison.png')
    print("Performance plot saved as 'performance_comparison.png'")
    
    # Wait time distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist([p.wait_time for p in fcfs_results], alpha=0.5, label='FCFS', color='#3498db')
    plt.hist([p.wait_time for p in rr_results], alpha=0.5, label='RR', color='#e74c3c')
    if rl_wait > 0:
        plt.hist([p.wait_time for p in rl_results], alpha=0.5, label='RL', color='#2ecc71')
    
    plt.xlabel('Wait Time')
    plt.ylabel('Frequency')
    plt.title('Wait Time Distribution')
    plt.legend()
    plt.savefig('wait_time_distribution.png')
    print("Wait time distribution plot saved as 'wait_time_distribution.png'")

if __name__ == "__main__":
    compare_and_plot()
