# Snake AI with Deep Q-Learning 🐍

An advanced implementation of a Snake game AI that learns through Deep Q-Learning, featuring real-time visualization, multi-agent support, and performance optimizations.

## 📚 Table of Contents

1. [🚀 Key Features](#-key-features)
2. [🛠️ Technical Specifications](#️-technical-specifications)
3. [📊 Real-time Visualization](#-real-time-visualization)
4. [🚀 Installation](#-installation)
5. [💻 Usage](#-usage)
6. [⚙️ Performance Features](#-performance-features)
7. [🔄 Automatic Saving](#-automatic-saving)
8. [🎮 Controls](#-controls)
9. [📈 Training Graphs](#-training-graphs)
10. [🎯 Latest Improvements](#-latest-improvements)
11. [🚀 Next Steps](#-next-steps)
12. [📝 Requirements](#-requirements)
13. [🔧 Configuration](#-configuration)
14. [📚 Project Structure](#-project-structure)
15. [🤝 Contributing](#-contributing)
16. [📄 License](#-license)

## 🚀 Key Features

- 🧠 Deep Q-Network with Priority Experience Replay (PER)
- 🎮 Real-time game visualization with informative HUD
- 📊 Multi-agent support (2-6 snakes training simultaneously)
- 📊 Live training statistics and performance graphs
- 🔄 Automatic checkpointing and model saving
- ⚡ CUDA-accelerated training with AMP
- 📈 Adaptive learning parameters
- 🎯 Enhanced exploration/exploitation balance
- 🤖 Interactive agent count configuration

## 🛠️ Technical Specifications

| Feature          | Description                                      |
|------------------|--------------------------------------------------|
| **Framework**    | PyTorch with CUDA support                        |
| **Visualization**| Pygame, Matplotlib                               |
| **Neural Network**| Input Layer: 17 neurons (enhanced state)        |
|                  | Hidden Layer: 512 neurons                        |
|                  | Output Layer: 4 actions (movement directions)    |
| **Training Parameters**| Learning Rate: 0.0005                      |
|                  | Gamma: 0.99                                      |
|                  | Initial Epsilon: 1.0                             |
|                  | Epsilon Decay: 0.997                             |
|                  | Memory Size: 100,000                             |
|                  | Batch Size: 64                                   |

## 📊 Real-time Visualization

- Individual scores for each snake
- Best score achieved
- Rolling average (last 100 games)
- Exploration rate (Epsilon)
- Samples collected
- Training performance (FPS)
- Loss function evolution
- Score distribution histogram
- Multi-agent interaction visualization

## 🚀 Installation

1. Clone the repository
```bash
git clone https://github.com/Anroshka/snake-ai.git
cd snake-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For CUDA support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 💻 Usage

Start training:
```bash
python train_multi.py
```

You'll be prompted to enter the number of snakes (2-6 recommended) for training.

## ⚙️ Performance Features

- Multi-agent learning environment
- Multi-threaded processing
- Automatic device selection (GPU/CPU)
- Optimized rendering with FPS control
- Gradient clipping for stability
- Automatic Mixed Precision (AMP)
- Priority Experience Replay
- Efficient memory management
- Advanced reward system

## 🔄 Automatic Saving

- Best model preservation
- Checkpoints every 10 episodes
- Training statistics graphs
- Performance metrics tracking
- Individual agent models saving

## 🎮 Controls

- ESC: Exit training
- Automatic gameplay during training
- Visualization every 10 episodes
- Training stats display
- Interactive agent count selection

## 📈 Training Graphs

- Multi-agent learning progress
- Loss function
- Epsilon decay
- Score distribution
- Moving averages
- Agent interaction patterns

## 🎯 Latest Improvements

- Added multi-agent support (2-6 snakes)
- Enhanced visualization with semi-transparent HUD
- Stabilized FPS for better visualization
- Minimum samples threshold for training start
- Improved error handling and stability
- Better memory management
- Adaptive learning parameters
- Interactive configuration system

## 🚀 Next Steps

- [ ] Add competitive and cooperative training modes
- [ ] Implement agent specialization
- [ ] Optimize multi-agent interactions
- [ ] Develop adaptive learning rate scheduling
- [ ] Explore different state representations
- [ ] Add agent personality traits

## 📝 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+
- Pygame 2.4.0
- Matplotlib for visualization

## 🔧 Configuration

All training parameters can be adjusted in `train_multi.py`:
- Number of agents (2-6)
- Episode count
- Memory size
- Batch size
- Learning rate
- Epsilon decay
- Visualization frequency
- Reward system parameters

## 📚 Project Structure

- `game.py`: Snake game environment with multi-agent support
- `model.py`: DQN implementation with PER
- `train_multi.py`: Multi-agent training loop and visualization
- `models/`: Saved models and checkpoints

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
