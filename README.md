# 🔥 AI-OS Scheduler: RL-Based CPU Scheduling

Replacing traditional CPU scheduling algorithms (FCFS, Round Robin) with a Reinforcement Learning agent developed using Gymnasium and Stable Baselines3.

## 🚀 Concept
This project explores the feasibility of using RL agents to manage real OS processes. By monitoring process metrics (CPU time, wait time, etc.) in real-time, the agent learns to prioritize tasks to minimize average wait time and maximize throughput.

## 📁 Project Structure
- `src/core/`: Real process management using `psutil` and `subprocess`.
- `src/env/`: Gymnasium environment for the RL agent.
- `src/schedulers/`: Implementation of baseline (FCFS, RR) and RL controllers.
- `scripts/`: Entry points for training and evaluating models.
- `results/`: Performance plots and visualization "figurines".

## 🛠️ Installation
1. **Clone the repository**:
   ```bash
   gh repo clone PiyushKBhattacharyya/AI-OS-Scheduler
   cd AI-OS-Scheduler
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage
### Training the RL Agent
Run the training script to train a PPO agent on a set of generated workloads:
```bash
python scripts/train.py
```

### Evaluation & Comparison
Compare the RL agent against traditional FCFS and Round Robin schedulers:
```bash
python scripts/evaluate.py
```
This will generate performance plots in the `results/` directory.

## 📊 Visual Interpretation
The project automatically saves "figurines" (plots) for:
- Average Wait Time Comparison
- Wait Time Distribution Histogram

## 📝 License
MIT License
