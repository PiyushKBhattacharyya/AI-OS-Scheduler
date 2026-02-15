import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from typing import List, Optional
from process_manager import ProcessManager, ManagedProcess

class RealProcessEnv(gym.Env):
    """
    Gymnasium environment for real-time CPU scheduling.
    Actions: Select a process to run for a short 'tick' duration.
    State: CPU time used, wait time, and remaining burst estimate.
    """
    def __init__(self, max_queue_size: int = 5, tick_duration: float = 0.5):
        super(RealProcessEnv, self).__init__()
        self.max_queue_size = max_queue_size
        self.tick_duration = tick_duration # Real-time seconds the process is resumed
        
        self.pm = ProcessManager()
        
        # Observation space: [cpu_time_used, burst_time, wall_wait_time]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.max_queue_size, 3), dtype=np.float32
        )
        
        # Action space: Index of process in queue to resume
        self.action_space = spaces.Discrete(self.max_queue_size)
        
        self.ready_queue: List[ManagedProcess] = []
        self.completed_processes: List[ManagedProcess] = []
        self.start_time = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.pm.cleanup()
        self.ready_queue = []
        self.completed_processes = []
        self.start_time = time.time()
        
        # Spawn some initial processes
        n_processes = options.get('n_processes', 3) if options else 3
        for i in range(n_processes):
            burst = np.random.uniform(1.0, 3.0) # 1-3 seconds of CPU burst
            self.ready_queue.append(self.pm.spawn_workload(i, burst))
            
        return self._get_obs(), {}

    def step(self, action):
        # 1. Update queue (check for finished processes)
        finished_this_step = []
        active_queue = []
        for p in self.ready_queue:
            if p.check_finished():
                finished_this_step.append(p)
                self.completed_processes.append(p)
            else:
                active_queue.append(p)
        self.ready_queue = active_queue

        if not self.ready_queue:
            return self._get_obs(), 0, True, False, {}

        # 2. Apply action: Resume selected, Suspend others
        reward = 0
        if action < len(self.ready_queue):
            # Suspend all
            for p in self.ready_queue:
                p.suspend()
            
            # Resume selected
            target = self.ready_queue[action]
            target.resume()
            
            # Wait for tick duration (actual execution time)
            time.sleep(self.tick_duration)
            
            # Suspend again after tick
            target.suspend()
            
            # Reward: Negative of average wait time in queue
            actual_wait = sum(time.time() - p.start_wall_time for p in self.ready_queue) / len(self.ready_queue)
            reward = -actual_wait
        else:
            # Invalid action – nothing runs for this tick
            time.sleep(self.tick_duration)
            reward = -10 # Penalty for idle CPU

        # 3. Final observation and termination
        terminated = len(self.ready_queue) == 0
        if terminated:
            # Bonus for completion
            total_turnaround = sum(p.completion_time - p.start_wall_time for p in self.completed_processes)
            reward += 50 - total_turnaround 

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        obs = np.zeros((self.max_queue_size, 3), dtype=np.float32)
        now = time.time()
        for i, p in enumerate(self.ready_queue[:self.max_queue_size]):
            cpu_used = p.get_cpu_time()
            wait_time = now - p.start_wall_time - cpu_used
            obs[i] = [cpu_used, p.burst_time, wait_time]
        return obs

    def close(self):
        self.pm.cleanup()
