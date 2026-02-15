import time
import os
import sys

# Add project root to path for modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.process_manager import ProcessManager

class Scheduler:
    """Base class for real-time CPU schedulers."""
    def __init__(self, n_processes=5, burst_range=(1.0, 3.0)):
        self.pm = ProcessManager()
        self.n_processes = n_processes
        self.burst_range = burst_range
        self.processes = []

    def setup(self):
        self.pm.cleanup()
        self.processes = []
        for i in range(self.n_processes):
            import numpy as np
            burst = np.random.uniform(*self.burst_range)
            self.processes.append(self.pm.spawn_workload(i, burst))

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")

class FCFS(Scheduler):
    """Real-time First-Come, First-Served Scheduler."""
    def run(self):
        print("Starting Real FCFS...")
        self.setup()
        
        for p in self.processes:
            print(f"Running {p.name} (Burst: {p.burst_time:.2f}s)")
            p.resume()
            while not p.check_finished():
                time.sleep(0.1)
            print(f"Finished {p.name}")
            
        return self.processes

class RR(Scheduler):
    """Real-time Round Robin Scheduler."""
    def __init__(self, n_processes=5, burst_range=(1.0, 3.0), quantum=0.5):
        super().__init__(n_processes, burst_range)
        self.quantum = quantum

    def run(self):
        print(f"Starting Real Round Robin (Quantum: {self.quantum}s)...")
        self.setup()
        
        queue = list(self.processes)
        completed = []
        
        while queue:
            p = queue.pop(0)
            if p.check_finished():
                completed.append(p)
                continue
                
            p.resume()
            time.sleep(self.quantum)
            p.suspend()
            
            if p.check_finished():
                completed.append(p)
            else:
                queue.append(p)
                
        return completed
