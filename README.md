# 🏁 CTF RL Project

A multi-agent reinforcement learning (MARL) project simulating a Capture-the-Flag (CTF) environment using advanced RL algorithms like DQN, PPO, MAPPO, and MADDPG. The project provides a competitive environment for evaluating coordination, communication, and strategic learning among agents.

## 📁 Project Structure

```
ctf_rl_project/
├── pycache/ # Compiled bytecode files
├── logs/ # Training logs and model checkpoints
├── anacopy.py # Model cloning or architecture utility
├── analyze_results.py # Evaluation and result plotting
├── buffer.py # Replay buffer implementation
├── commnet.py # Agent communication neural network
├── ctf_env.py # Custom CTF environment
├── dqn.py # DQN algorithm
├── logger.py # Logging utility
├── maddpg.py # MADDPG algorithm
├── mappo.py # MAPPO algorithm
├── play.py # Run trained agents
├── ppo.py # PPO algorithm
├── requirements.txt # Dependencies
├── tarmac.py # Map and agent placement logic
├── train_visual.py # Visualized training
├── train.py # Main training script
├── visualization.py # Training visualization tools
└── wrappers.py # Gym wrappers
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ctf_rl_project.git
cd ctf_rl_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** Python 3.8+ is recommended. You may need additional packages such as gym, pygame, numpy, and matplotlib.

## 🧠 Algorithms Implemented

- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)
- Multi-Agent PPO (MAPPO)
- Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
- Communication Networks (CommNet)

## 🏗️ Environment Description

A custom 15×15 grid world simulating a competitive Capture-the-Flag game between two teams:
- Red Team vs. Blue Team
- Objective: Capture opponent's flag & defend own
- Obstacles and terrains add strategy
- Partial observability for realistic scenarios
- Action Space: {Stay, Up, Down, Left, Right} per agent
- Observation Space: Local 5×5 grid patches per agent

## 🏋️‍♂️ Training Agents

### Basic Training
```bash
python train.py --algo ppo --env ctf
```

### Visualized Training
```bash
python train_visual.py --algo maddpg
```

## 🧪 Evaluation

### Run Trained Agent
```bash
python play.py --model checkpoints/ppo_final.pth --env ctf
```

### Analyze Results
```bash
python analyze_results.py
```

## 📊 Visualization

Generate reward plots, success graphs, and other metrics:
```bash
python visualization.py --logdir logs/mappo/
```

## 🔍 Logging

Training logs include:
- Episode reward history
- Win/loss outcomes
- Loss trends
- Model checkpoints

All saved in the `logs/` directory and structured via `logger.py`.

## 🧑‍💻 Contributors

- Bandaru Ritesh Kumar (CS22B2043)
- Ganesh Banotu (CS22B2008)

## 📄 License

This project is released under the MIT License.
(Or replace with your own license.)

## 🙏 Acknowledgements

This project was developed as part of CS3009 - Reinforcement Learning at IIITDM Kancheepuram, under the guidance of Dr. Rahul Raman.

## 🌐 References

- MADDPG Paper (Lowe et al., 2017)
- CommNet Paper (Sukhbaatar et al., 2016)
- MAPPO Paper (Yu et al., 2022)
