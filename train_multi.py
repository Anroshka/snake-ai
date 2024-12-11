import pygame
import numpy as np
from game import SnakeGame
from model import MultiAgentDQN
import matplotlib.pyplot as plt
from collections import deque
import time
import os

def plot_scores(scores_history, avg_scores_history, filename='training_stats_multi.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(scores_history, label='Scores', alpha=0.4)
    plt.plot(avg_scores_history, label='Average Scores', linewidth=2)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def get_valid_input(prompt, min_val, max_val):
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")

def train():
    # Интерактивный ввод количества змеек
    print("\n=== Snake AI Training Configuration ===")
    print("How many snakes would you like to train? (2-6 recommended)")
    n_agents = get_valid_input("Enter number of snakes: ", 2, 6)
    
    print(f"\nInitializing training with {n_agents} snakes...")
    print("(Press Ctrl+C to stop training at any time)")
    
    n_games = 1000
    
    # Инициализация игры и агента с улучшенными параметрами
    game = SnakeGame(width=640, height=480, cell_size=20, n_agents=n_agents)
    agent = MultiAgentDQN(
        n_agents=n_agents,
        input_size=17,
        hidden_size=512,
        output_size=4,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.997,
        memory_size=100000,
        batch_size=64
    )
    
    scores_history = []
    avg_scores_history = []
    best_avg_score = float('-inf')
    scores_window = deque(maxlen=100)
    
    # Создаем директорию для моделей, если её нет
    if not os.path.exists('models'):
        os.makedirs('models')
    
    try:
        print("\nTraining started!")
        print("Visual preview will be shown every 10 episodes")
        print("Training statistics will be saved in 'training_stats_multi.png'")
        print("\nPress Ctrl+C to stop training and save the current model")
        
        for episode in range(n_games):
            states = game.reset()
            episode_scores = [0] * n_agents
            done = False
            steps = 0
            
            # Флаг для отображения этого эпизода
            should_render = episode % 10 == 0
            
            if should_render:
                print(f"\nVisualizing episode {episode + 1}")
            
            while not done and steps < 1000:
                steps += 1
                
                # Обработка событий Pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                # Выбираем действия для всех агентов
                actions = agent.select_actions(states)
                
                # Выполняем действия в игре
                next_states, rewards, done = game.step(actions)
                
                # Сохраняем опыт
                agent.store_transitions(states, actions, rewards, next_states, done)
                
                # Обновляем состояния и счет
                states = next_states
                for i in range(n_agents):
                    episode_scores[i] += rewards[i]
                
                # Обучаем агентов
                loss = agent.train()
                
                # Обновляем целевые сети каждые 50 шагов
                if steps % 50 == 0:
                    agent.update_target_networks()
                
                # Отрисовываем игру только каждые 10 эпизодов
                if should_render:
                    game.render()
                    time.sleep(0.05)
            
            # Сохраняем статистику
            avg_episode_score = np.mean(episode_scores)
            scores_window.append(avg_episode_score)
            scores_history.append(avg_episode_score)
            avg_score = np.mean(scores_window)
            avg_scores_history.append(avg_score)
            
            # Выводим прогресс с дополнительной информацией
            print(f'Episode: {episode + 1}, Steps: {steps}, '
                  f'Scores: [{", ".join([f"{score:.1f}" for score in episode_scores])}], '
                  f'Avg Score: {avg_score:.2f}, Epsilon: {agent.agents[0].epsilon:.3f}')
            
            # Сохраняем лучшую модель
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save_models('models/best_model')
                print(f'New record! Average score: {avg_score:.2f}')
            
            # Каждые 100 эпизодов обновляем график
            if (episode + 1) % 100 == 0:
                plot_scores(scores_history, avg_scores_history)
                print(f'\nTraining progress saved. Episode {episode + 1}')
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Сохраняем финальную модель
        print("Saving final model...")
        agent.save_models('models/final_model')
        plot_scores(scores_history, avg_scores_history)
        
        # Закрываем игру
        game.close()
        print("Training completed!")

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.") 