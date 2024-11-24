# Snake AI with Deep Q-Learning

An advanced implementation of a Snake game AI that learns through Deep Q-Learning, featuring real-time visualization and performance optimizations.

## 🚀 Key Features

- 🧠 Deep Q-Network with Priority Experience Replay (PER)
- 🎮 Real-time game visualization with informative HUD
- 📊 Live training statistics and performance graphs
- 🔄 Automatic checkpointing and model saving
- ⚡ CUDA-accelerated training with AMP
- 📈 Adaptive learning parameters
- 🎯 Enhanced exploration/exploitation balance

## 🛠️ Technical Specifications

- **Framework**: PyTorch with CUDA support
- **Visualization**: Pygame, Matplotlib, Seaborn
- **Neural Network**: 
  - Input Layer: 12 neurons (state representation)
  - Hidden Layer: 256 neurons
  - Output Layer: 4 actions (movement directions)
- **Training Parameters**:
  - Learning Rate: 0.001
  - Gamma: 0.99
  - Initial Epsilon: 1.0
  - Epsilon Decay: 0.995
  - Memory Size: 100,000
  - Batch Size: 128

## 📊 Real-time Visualization

- Current game score
- Best score achieved
- Rolling average (last 100 games)
- Exploration rate (Epsilon)
- Samples collected
- Training performance (FPS)
- Loss function evolution
- Score distribution histogram

## 🚀 Installation

1. Clone the repository
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
python train.py
```

## ⚙️ Performance Features

- Multi-threaded processing
- Automatic device selection (GPU/CPU)
- Optimized rendering with FPS control
- Gradient clipping for stability
- Automatic Mixed Precision (AMP)
- Priority Experience Replay
- Efficient memory management

## 🔄 Automatic Saving

- Best model preservation
- Checkpoints every 10 episodes
- Training statistics graphs
- Performance metrics tracking

## 🎮 Controls

- ESC: Exit training
- Automatic gameplay during training
- Visualization every 10 episodes
- Training stats display

## 📈 Training Graphs

- Learning progress
- Loss function
- Epsilon decay
- Score distribution
- Moving averages

## 🎯 Latest Improvements

- Enhanced visualization with semi-transparent HUD
- Stabilized FPS for better visualization
- Minimum samples threshold for training start
- Improved error handling and stability
- Better memory management
- Adaptive learning parameters

## 📝 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+
- Pygame 2.4.0
- Matplotlib and Seaborn for visualization

## 🔧 Configuration

All training parameters can be adjusted in `train.py`:
- Episode count
- Memory size
- Batch size
- Learning rate
- Epsilon decay
- Visualization frequency

## 📚 Project Structure

- `game.py`: Snake game environment
- `model.py`: DQN implementation with PER
- `train.py`: Training loop and visualization
- `models/`: Saved models and checkpoints

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
