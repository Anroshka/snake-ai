import torch
import numpy as np
from game import SnakeGame
from model import QLearningAgent
import pygame
import time
from datetime import datetime
import os

def train():
    env = SnakeGame()
    agent = QLearningAgent()
    
    episodes = 1000
    max_steps = 1000
    target_update_freq = 5
    render_freq = 50  # Частота отображения игры
    save_freq = 100   # Частота сохранения модели
    
    best_score = 0
    scores = []
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
                    pygame.time.wait(50)
                    
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train()
                
                state = next_state
                total_reward += reward
                
                if reward == 10:  # Змейка съела еду
                    score += 1
                
                if done:
                    break
            
            episode_time = time.time() - episode_start_time
            scores.append(score)
            if score > best_score:
                best_score = score
                # Сохраняем лучшую модель
                agent.save_model(os.path.join(save_dir, "best_model.pth"))
            
            # Обновляем целевую сеть
            if episode % target_update_freq == 0:
                agent.update_target_network()
            
            # Сохраняем чекпоинт
            if episode % save_freq == 0:
                agent.save_model(os.path.join(save_dir, f"checkpoint_{episode}.pth"))
            
            # Вычисляем среднее значение за последние 100 эпизодов
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            
            # Вычисляем производительность
            steps_per_sec = (step + 1) / episode_time
            
            print(f"Episode {episode}, Score: {score}, Best: {best_score}, "
                  f"Avg100: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}, "
                  f"Steps/sec: {steps_per_sec:.1f}")
    
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    finally:
        total_time = time.time() - start_time
        pygame.quit()
        
        # Сохраняем финальную модель
        agent.save_model(os.path.join(save_dir, "final_model.pth"))
        
        print("\nСтатистика обучения:")
        print(f"Время обучения: {total_time/3600:.2f} часов")
        print(f"Лучший результат: {best_score}")
        print(f"Средний результат за последние 100 игр: {np.mean(scores[-100:]):.2f}")
        print(f"Модели сохранены в директории: {save_dir}")

if __name__ == "__main__":
    train()
