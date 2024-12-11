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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Определяет, насколько сильно приоритет влияет на выборку
        self.beta = beta    # Важность весов для коррекции смещения
        self.beta_increment = 0.001  # Постепенное увеличение beta до 1
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def store(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
        
        # Новый опыт получает максимальный приоритет
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None, None, None
        
        # Вычисляем вероятности выборки на основе приоритетов
        probs = self.priorities[:len(self.memory)]
        probs = probs ** self.alpha
        probs = probs / probs.sum()
        
        # Выбираем индексы на основе приоритетов
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Вычисляем веса важности
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), weights, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

class QLearningAgent:
    def __init__(self, input_size=12, hidden_size=256, output_size=4, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=100000,
                 batch_size=128):
        
        # Определяем устройство (CPU/CUDA)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Инициализация сетей
        self.policy_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')  # Используем reduction='none' для PER
        
        # Параметры обучения
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Память для опыта с приоритетами
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Включаем автоматическое смешанное вычисление точности только если есть CUDA
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def train(self):
        # Проверяем, достаточно ли опыта для обучения
        if len(self.memory.memory) < self.batch_size:
            return None
            
        result = self.memory.sample(self.batch_size)
        if result is None:
            return None
            
        batch, weights, indices = result
        states, actions, rewards, next_states, dones = batch
        
        # Преобразование в тензоры
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Обучение с или без AMP в зависимости от наличия CUDA
        if self.use_amp:
            with torch.cuda.amp.autocast():
                current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                
                # Вычисляем TD-ошибки и взвешенную функцию потерь
                td_errors = (current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
                loss = (self.criterion(current_q_values.squeeze(), target_q_values) * weights).mean()
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Вычисляем TD-ошибки и взвешенную функцию потерь
            td_errors = (current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
            loss = (self.criterion(current_q_values.squeeze(), target_q_values) * weights).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Обновляем приоритеты в памяти
        self.memory.update_priorities(indices, td_errors)
        
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

class MultiAgentDQN:
    def __init__(self, n_agents=4, input_size=17, hidden_size=256, output_size=4,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=100000,
                 batch_size=128):
        
        self.n_agents = n_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Создаем агентов
        self.agents = []
        for _ in range(n_agents):
            agent = QLearningAgent(input_size, hidden_size, output_size,
                                 learning_rate, gamma, epsilon,
                                 epsilon_min, epsilon_decay, memory_size,
                                 batch_size)
            self.agents.append(agent)
    
    def store_transitions(self, states, actions, rewards, next_states, dones):
        for i in range(self.n_agents):
            self.agents[i].store_transition(states[i], actions[i], rewards[i],
                                          next_states[i], dones)
    
    def select_actions(self, states):
        return [agent.select_action(state) for agent, state in zip(self.agents, states)]
    
    def train(self):
        losses = []
        for agent in self.agents:
            loss = agent.train()
            # Добавляем loss только если обучение действительно произошло
            if loss is not None:
                losses.append(loss)
        # Возвращаем среднее значение loss только если были успешные обучения
        return np.mean(losses) if losses else None
    
    def update_target_networks(self):
        for agent in self.agents:
            agent.update_target_network()
    
    def save_models(self, path):
        """Сохраняет модели всех агентов"""
        for i, agent in enumerate(self.agents):
            agent_path = f"{path}_agent_{i}"
            agent.save_model(agent_path)
    
    def load_models(self, path):
        """Загружает модели всех агентов"""
        success = True
        for i, agent in enumerate(self.agents):
            agent_path = f"{path}_agent_{i}"
            if not agent.load_model(agent_path):
                success = False
        return success
