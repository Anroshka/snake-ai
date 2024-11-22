# Snake AI Learning Game

An AI-powered Snake game that learns and improves through deep reinforcement learning using PyTorch.

## Features

- Deep Q-Learning implementation with PyTorch
- CUDA support for GPU acceleration
- Real-time visualization of the learning process
- Automatic model checkpointing and saving of best models
- Comprehensive training statistics display
- Configurable hyperparameters
- Smooth game animations with Pygame

## Requirements

- Python 3.10
- PyTorch (with CUDA support recommended)
- Pygame 2.4.0
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/snake-ai.git
cd snake-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. If you have an NVIDIA GPU, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

1. Start training:
```bash
python train.py
```

2. Controls during training:
- ESC: Stop training
- Close window: Stop training

## Project Structure

- `game.py`: Snake game environment implementation
- `model.py`: Deep Q-Network and training agent implementation
- `train.py`: Training loop and visualization
- `models/`: Directory for saved model checkpoints
  - `best_model.pth`: Best performing model
  - `checkpoint_X.pth`: Regular checkpoints every 100 episodes

## Technical Details

### Neural Network Architecture
- Input: 12 feature state representation
- Hidden layers: Two-layer architecture
- Output: Q-values for possible actions

### Training Parameters
- Learning Rate: 0.001
- Discount Factor (gamma): 0.95
- Initial Exploration Rate (epsilon): 1.0
- Epsilon Decay: 0.997
- Minimum Epsilon: 0.01
- Replay Memory Size: Based on available system memory
- Batch Size: Configurable

### Performance Features
- Automatic CUDA detection and utilization
- Gradient clipping for training stability
- Experience replay with deque
- Target network for stable learning
- Automatic Mixed Precision (AMP) training when CUDA is available

## License

MIT License - feel free to use this code for your own projects.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
