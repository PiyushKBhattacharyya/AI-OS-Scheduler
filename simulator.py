import dataclasses
from typing import List

@dataclasses.dataclass
class Process:
    pid: int
    arrival_time: float
    burst_time: float
    remaining_time: float = 0.0
    start_time: float = -1.0
    end_time: float = -1.0
    wait_time: float = 0.0
    turnaround_time: float = 0.0

    def __post_init__(self):
        self.remaining_time = self.burst_time

class Simulator:
    """Base class for CPU simulators."""
    def __init__(self, processes: List[Process]):
        self.processes = sorted([dataclasses.replace(p) for p in processes], key=lambda x: x.arrival_time)
        self.current_time = 0.0
        self.completed_processes = []

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")

class FCFSSimulator(Simulator):
    """First-Come, First-Served Scheduler."""
    def run(self):
        for p in self.processes:
            if self.current_time < p.arrival_time:
                self.current_time = p.arrival_time
            
            p.start_time = self.current_time
            p.wait_time = p.start_time - p.arrival_time
            self.current_time += p.burst_time
            p.end_time = self.current_time
            p.turnaround_time = p.end_time - p.arrival_time
            p.remaining_time = 0
            self.completed_processes.append(p)
        return self.completed_processes

class RRSimulator(Simulator):
    """Round Robin Scheduler."""
    def __init__(self, processes: List[Process], quantum: float = 2.0):
        super().__init__(processes)
        self.quantum = quantum

    def run(self):
        queue = []
        process_idx = 0
        n = len(self.processes)
        
        while process_idx < n or queue:
            # Add processes that have arrived
            while process_idx < n and self.processes[process_idx].arrival_time <= self.current_time:
                queue.append(self.processes[process_idx])
                process_idx += 1
            
            if not queue:
                if process_idx < n:
                    self.current_time = self.processes[process_idx].arrival_time
                    continue
                else:
                    break
            
            p = queue.pop(0)
            if p.start_time == -1:
                p.start_time = self.current_time
            
            exec_time = min(p.remaining_time, self.quantum)
            p.remaining_time -= exec_time
            self.current_time += exec_time
            
            # Add processes that arrived during execution
            while process_idx < n and self.processes[process_idx].arrival_time <= self.current_time:
                queue.append(self.processes[process_idx])
                process_idx += 1
            
            if p.remaining_time > 0:
                queue.append(p)
            else:
                p.end_time = self.current_time
                p.turnaround_time = p.end_time - p.arrival_time
                p.wait_time = p.turnaround_time - p.burst_time
                self.completed_processes.append(p)
                
        return self.completed_processes
