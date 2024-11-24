import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

# Оптимизации CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

# Проверяем доступность CUDA и выводим информацию
print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class QLearningAgent:
    def __init__(self, input_size=12, hidden_size=256, output_size=4, 
                 learning_rate=0.001, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.997, memory_size=100000,
                 batch_size=64):
        
        # Определяем устройство (CPU/CUDA)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Инициализация сетей
        self.policy_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Параметры обучения
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Память для опыта
        self.memory = deque(maxlen=memory_size)
        
        # Включаем автоматическое смешанное вычисление точности только если есть CUDA
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return None
            
        # Выборка мини-батча
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Преобразование в тензоры
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Обучение с или без AMP в зависимости от наличия CUDA
        if self.use_amp:
            with torch.cuda.amp.autocast():
                current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                loss = self.criterion(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Обычное обучение без AMP
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = self.criterion(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Обновляем epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, path):
        """Сохраняет модель и состояние обучения"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'device': str(self.device)
        }
        torch.save(checkpoint, path)
        
    def load_model(self, path):
        """Загружает модель и состояние обучения"""
        if not os.path.exists(path):
            print(f"Модель не найдена: {path}")
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        # Переносим модели на правильное устройство после загрузки
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        return True
