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

def plot_stats(scores, avg_scores, losses, epsilon_history):
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
    plt.plot(losses, label='Loss', color='orange')
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # График epsilon
    plt.subplot(2, 2, 3)
    plt.plot(epsilon_history, label='Epsilon', color='green')
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)
    
    # Гистограмма распределения счета
    plt.subplot(2, 2, 4)
    plt.hist(scores, bins=30, color='purple', alpha=0.7)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_stats.png')
    plt.close()

def train():
    env = SnakeGame()
    agent = QLearningAgent()
    
    episodes = 1000
    max_steps = 1000
    target_update_freq = 5
    render_freq = 10  # Показываем игру каждые 10 эпизодов
    save_freq = 100   # Частота сохранения модели
    
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
                    info_surface = pygame.Surface((200, 150))
                    info_surface.fill((50, 50, 50))
                    
                    texts = [
                        f"Episode: {episode}",
                        f"Score: {score}",
                        f"Best: {best_score}",
                        f"Epsilon: {agent.epsilon:.2f}"
                    ]
                    
                    for i, text in enumerate(texts):
                        text_surface = font.render(text, True, (255, 255, 255))
                        info_surface.blit(text_surface, (10, 10 + i * 30))
                    
                    env.screen.blit(info_surface, (env.width - 210, 10))
                    pygame.display.flip()
                    pygame.time.wait(50)
                    
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
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
            scores.append(score)
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
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
                plot_stats(scores, avg_scores, losses, epsilon_history)
            
            steps_per_sec = (step + 1) / episode_time
            print(f"Episode {episode}, Score: {score}, Best: {best_score}, "
                  f"Avg100: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}, "
                  f"Steps/sec: {steps_per_sec:.1f}")
    
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    finally:
        total_time = time.time() - start_time
        pygame.quit()
        
        # Сохраняем финальную модель и графики
        agent.save_model(os.path.join(save_dir, "final_model.pth"))
        plot_stats(scores, avg_scores, losses, epsilon_history)
        
        print("\nСтатистика обучения:")
        print(f"Время обучения: {total_time/3600:.2f} часов")
        print(f"Лучший результат: {best_score}")
        print(f"Средний результат за последние 100 игр: {np.mean(scores[-100:]):.2f}")
        print(f"Модели сохранены в директории: {save_dir}")
        print(f"Графики сохранены в файле: training_stats.png")

if __name__ == "__main__":
    train()
