import time
import sys
import argparse

def heavy_computation(duration):
    """A CPU-intensive task."""
    start_time = time.time()
    counter = 0
    # Perform math to keep CPU busy
    while time.time() - start_time < duration:
        _ = sum(i * i for i in range(1000))
        counter += 1
        # Optional: yield to OS or report progress occasionally
    return counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--burst", type=float, default=5.0)
    args = parser.parse_args()

    # print(f"Process {args.id} started. Target burst: {args.burst}s")
    
    # We use CPU time instead of wall time to measure work
    # But for simplicity in this script, we just run until we hit the 'burst' 
    # of wall clock time if we were the only process running.
    # A better way is to measure process CPU time.
    
    import psutil
    p = psutil.Process()
    start_cpu = p.cpu_times().user + p.cpu_times().system
    
    while (p.cpu_times().user + p.cpu_times().system) - start_cpu < args.burst:
        # Do work
        _ = 2**10000 
        
    # print(f"Process {args.id} finished.")
