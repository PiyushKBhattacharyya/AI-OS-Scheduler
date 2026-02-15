import subprocess
import psutil
import time
import os
import sys

class ManagedProcess:
    def __init__(self, pid, name, workload_id, burst_time):
        self.pid = pid
        self.name = name
        self.workload_id = workload_id
        self.burst_time = burst_time
        self.handle = psutil.Process(pid)
        self.start_wall_time = time.time()
        self.completion_time = None
        self.is_finished = False
        self.last_cpu_time = 0.0

    def get_cpu_time(self):
        try:
            times = self.handle.cpu_times()
            self.last_cpu_time = times.user + times.system
            return self.last_cpu_time
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return self.last_cpu_time

    def get_status(self):
        try:
            return self.handle.status()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "finished"

    def suspend(self):
        try:
            if self.handle.status() != psutil.STATUS_STOPPED:
                self.handle.suspend()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def resume(self):
        try:
            if self.handle.status() == psutil.STATUS_STOPPED:
                self.handle.resume()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def check_finished(self):
        if self.is_finished:
            return True
        
        # Check if process is still running or finished its CPU burst
        if not self.handle.is_running() or self.get_cpu_time() >= self.burst_time:
            self.is_finished = True
            self.completion_time = time.time()
            self.terminate()
            return True
        return False

    def terminate(self):
        try:
            self.handle.terminate()
        except:
            pass

class ProcessManager:
    def __init__(self):
        self.processes = []
        self.python_exe = sys.executable

    def spawn_workload(self, workload_id, burst_time):
        cmd = [self.python_exe, "workload_task.py", "--id", str(workload_id), "--burst", str(burst_time)]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        managed = ManagedProcess(p.pid, f"workload_{workload_id}", workload_id, burst_time)
        # Initially suspend all spawned processes to let the scheduler decide
        managed.suspend()
        self.processes.append(managed)
        return managed

    def get_active_processes(self):
        self.processes = [p for p in self.processes if not p.check_finished()]
        return self.processes

    def cleanup(self):
        for p in self.processes:
            p.terminate()
        self.processes = []
