import torch
import numpy as np
from game import SnakeGame
from model import QLearningAgent
import pygame
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import argparse

def plot_stats(scores, avg_scores, losses, epsilon_history):
    if not scores:  # Если нет данных, не создаем график
        return
        
    plt.figure(figsize=(15, 10))
    
    # График счета
    plt.subplot(2, 2, 1)
    plt.plot(scores, label='Score', alpha=0.4)
    plt.plot(avg_scores, label='Average Score', color='red')
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # График потерь
    plt.subplot(2, 2, 2)
    if losses:  # Проверяем, есть ли данные о потерях
        plt.plot(losses, label='Loss', color='orange')
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # График epsilon
    plt.subplot(2, 2, 3)
    if epsilon_history:  # Проверяем, есть ли данные об epsilon
        plt.plot(epsilon_history, label='Epsilon', color='green')
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.legend()
        plt.grid(True)
    
    # Гистограмма распределения счета
    plt.subplot(2, 2, 4)
    if len(scores) > 1:  # Нужно минимум 2 значения для гистограммы
        plt.hist(scores, bins=min(30, len(scores)), color='purple', alpha=0.7)
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_stats.png')
    plt.close()

def create_heatmap(positions, width, height, cell_size):
    grid = np.zeros((height//cell_size, width//cell_size))
    for pos in positions:
        x, y = pos
        grid[y][x] += 1
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid, cmap='YlOrRd', annot=False)
    plt.title('Snake Movement Heatmap')
    plt.savefig('heatmap.png')
    plt.close()

def train():
    env = SnakeGame()
    agent = QLearningAgent()
    
    episodes = 1000
    max_steps = 1000
    target_update_freq = 5
    render_freq = 10  # Показываем игру каждые 10 эпизодов
    save_freq = 10    # Сохраняем чекпоинт каждые 10 эпизодов
    min_samples_for_training = 1000  # Минимальное количество сэмплов для начала обучения
    
    # Отключаем синхронизацию CUDA для ускорения
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Включаем многопоточность для CPU
    torch.set_num_threads(4)
    
    best_score = 0
    scores = []
    avg_scores = []
    losses = []
    epsilon_history = []
    running = True
    samples_collected = 0
    avg_score = 0.0  # Инициализируем среднее значение
    
    # Создаем директорию для сохранения моделей
    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    start_time = time.time()
    try:
        for episode in range(episodes):
            if not running:
                break
                
            state = env.reset()
            total_reward = 0
            score = 0
            episode_start_time = time.time()
            episode_losses = []
            
            # Отображаем каждые render_freq эпизодов
            render = episode % render_freq == 0
            
            for step in range(max_steps):
                # Обработка событий Pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                
                if not running:
                    break
                
                if render:
                    env.render()
                    
                    # Добавляем информацию на экран
                    font = pygame.font.Font(None, 36)
                    info_surface = pygame.Surface((300, 200))
                    info_surface.fill((50, 50, 50))
                    info_surface.set_alpha(230)  # Полупрозрачность
                    
                    texts = [
                        f"Episode: {episode:4d}",
                        f"Score: {score:3d}",
                        f"Best: {best_score:3d}",
                        f"Avg100: {avg_score:6.2f}",
                        f"Epsilon: {agent.epsilon:.3f}",
                        f"Samples: {samples_collected:6d}"
                    ]
                    
                    y_offset = 10
                    for text in texts:
                        text_surface = font.render(text, True, (255, 255, 255))
                        info_surface.blit(text_surface, (10, y_offset))
                        y_offset += 30
                    
                    env.screen.blit(info_surface, (env.width - 310, 10))
                    pygame.display.flip()
                    
                    # Контролируем FPS для визуализации
                    if step == 0:  # В начале эпизода делаем паузу для лучшей видимости
                        pygame.time.wait(500)
                    else:
                        pygame.time.wait(50)  # ~20 FPS
                
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                samples_collected += 1
                
                if samples_collected >= min_samples_for_training:
                    loss = agent.train()
                    if loss is not None:
                        episode_losses.append(loss)
                
                state = next_state
                total_reward += reward
                
                if reward == 10:  # Змейка съела еду
                    score += 1
                
                if done:
                    break
            
            episode_time = time.time() - episode_start_time
            steps_per_sec = (step + 1) / max(episode_time, 0.001)  # Защита от деления на ноль
            
            scores.append(score)
            
            # Вычисляем среднее значение за последние 100 эпизодов
            if len(scores) >= 100:
                avg_score = np.mean(scores[-100:])
            else:
                avg_score = np.mean(scores)
            
            avg_scores.append(avg_score)
            
            if episode_losses:
                losses.extend(episode_losses)
            epsilon_history.append(agent.epsilon)
            
            if score > best_score:
                best_score = score
                agent.save_model(os.path.join(save_dir, "best_model.pth"))
            
            if episode % target_update_freq == 0:
                agent.update_target_network()
            
            if episode % save_freq == 0:
                agent.save_model(os.path.join(save_dir, f"checkpoint_{episode}.pth"))
                # Обновляем графики каждые save_freq эпизодов
                if scores:  # Проверяем, есть ли данные для графиков
                    plot_stats(scores, avg_scores, losses, epsilon_history)
            
            # Форматируем вывод статистики
            print(f"Episode {episode:4d} | Score: {score:3d} | Best: {best_score:3d} | "
                  f"Avg100: {avg_score:6.2f} | Eps: {agent.epsilon:.3f} | "
                  f"Steps/s: {steps_per_sec:7.1f} | Samples: {samples_collected:6d}")
            
            # Небольшая задержка для стабильности
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    finally:
        total_time = time.time() - start_time
        pygame.quit()
        
        # Сохраняем финальную модель и графики
        agent.save_model(os.path.join(save_dir, "final_model.pth"))
        if scores:  # Проверяем, есть ли данные для графиков
            plot_stats(scores, avg_scores, losses, epsilon_history)
        
        print("\nСтатистика обучения:")
        print(f"Время обучения: {total_time/3600:.2f} часов")
        print(f"Лучший результат: {best_score}")
        print(f"Средний результат за последние 100 игр: {np.mean(scores[-100:]):.2f}")
        print(f"Модели сохранены в директории: {save_dir}")
        print(f"Графики сохранены в файле: training_stats.png")

def demonstrate_best_model(model_path, episodes=5, delay=100):
    env = SnakeGame()
    agent = QLearningAgent()
    
    if not agent.load_model(model_path):
        print(f"Не удалось загрузить модель из {model_path}")
        return
    
    agent.epsilon = 0  # Отключаем исследование
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        positions = []  # Для тепловой карты
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            env.render()
            positions.append(env.snake[0])  # Записываем позицию головы змеи
            
            # Показываем Q-значения
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor)[0].cpu().numpy()
            
            # Отображаем Q-значения на экране
            font = pygame.font.Font(None, 36)
            actions = ['←', '→', '↑', '↓']
            for i, (action, q_val) in enumerate(zip(actions, q_values)):
                text = f"{action}: {q_val:.2f}"
                text_surface = font.render(text, True, (255, 255, 255))
                env.screen.blit(text_surface, (env.width + 10, 50 + i * 30))
            
            pygame.display.flip()
            pygame.time.wait(delay)  # Замедляем для наблюдения
            
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        print(f"Эпизод {episode + 1}, Счет: {total_reward}")
        create_heatmap(positions, env.width, env.height, env.cell_size)
    
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Snake AI Training and Demo')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'demo'],
                      help='Mode: train or demo')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                      help='Path to model for demo mode')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    else:
        demonstrate_best_model(args.model)
