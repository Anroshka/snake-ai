import pygame
import numpy as np
import random

class SnakeGame:
    def __init__(self, width=640, height=480, cell_size=20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.n_cells_x = width // cell_size
        self.n_cells_y = height // cell_size
        
        # Цвета
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        # Инициализация Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.reset()
        
    def reset(self):
        # Начальное положение змейки
        self.snake = [(self.n_cells_x//2, self.n_cells_y//2)]
        self.direction = random.choice([(1,0), (0,1), (-1,0), (0,-1)])
        self.score = 0
        self.steps = 0
        self.food = self._place_food()
        self.game_over = False
        return self._get_state()
        
    def _place_food(self):
        while True:
            food = (random.randint(0, self.n_cells_x-1),
                   random.randint(0, self.n_cells_y-1))
            if food not in self.snake:
                return food
                
    def _get_state(self):
        head = self.snake[0]
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)
        
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)
        
        state = [
            # Опасность прямо
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),
            
            # Опасность справа
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),
            
            # Опасность слева
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # Направление движения
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Положение еды
            self.food[0] < head[0],  # еда слева
            self.food[0] > head[0],  # еда справа
            self.food[1] < head[1],  # еда сверху
            self.food[1] > head[1],  # еда снизу
            
            # Длина змейки
            len(self.snake) / (self.n_cells_x * self.n_cells_y)  # нормализованная длина
        ]
        
        return np.array(state, dtype=int)
        
    def _is_collision(self, point):
        # Проверяем столкновение со стенами
        if point[0] >= self.n_cells_x or point[0] < 0 or \
           point[1] >= self.n_cells_y or point[1] < 0:
            return True
        # Проверяем столкновение с телом змеи
        if point in self.snake[1:]:
            return True
        return False
        
    def step(self, action):
        self.steps += 1
        # Обновляем направление движения
        if action == 0:  # Прямо
            pass
        elif action == 1:  # Право
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:  # Лево
            self.direction = (-self.direction[1], self.direction[0])
            
        # Двигаем змейку
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Проверяем столкновение
        reward = 0
        self.game_over = False
        if self._is_collision(new_head):
            self.game_over = True
            reward = -10
        else:
            self.snake.insert(0, new_head)
            # Проверяем, съела ли змейка еду
            if new_head == self.food:
                self.score += 1
                reward = 10
                self.food = self._place_food()
            else:
                self.snake.pop()
                # Небольшой отрицательный reward за каждый шаг
                reward = -0.1
                
        return self._get_state(), reward, self.game_over
        
    def render(self):
        self.screen.fill(self.BLACK)
        
        # Рисуем сетку
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.width, y))
        
        # Рисуем змейку
        for i, segment in enumerate(self.snake):
            color = self.BLUE if i == 0 else self.GREEN
            rect = pygame.Rect(
                segment[0] * self.cell_size,
                segment[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, color, rect)
            
            # Добавляем глаза для головы змеи
            if i == 0:
                eye_size = self.cell_size // 4
                eye_offset = self.cell_size // 4
                
                # Левый глаз
                left_eye = pygame.Rect(
                    segment[0] * self.cell_size + eye_offset,
                    segment[1] * self.cell_size + eye_offset,
                    eye_size,
                    eye_size
                )
                # Правый глаз
                right_eye = pygame.Rect(
                    segment[0] * self.cell_size + self.cell_size - eye_offset - eye_size,
                    segment[1] * self.cell_size + eye_offset,
                    eye_size,
                    eye_size
                )
                pygame.draw.rect(self.screen, self.WHITE, left_eye)
                pygame.draw.rect(self.screen, self.WHITE, right_eye)
        
        # Рисуем еду
        food_rect = pygame.Rect(
            self.food[0] * self.cell_size,
            self.food[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.RED, food_rect)
        
        # Отображаем статистику
        score_text = self.font.render(f'Score: {self.score}', True, self.WHITE)
        steps_text = self.font.render(f'Steps: {self.steps}', True, self.WHITE)
        length_text = self.font.render(f'Length: {len(self.snake)}', True, self.WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 50))
        self.screen.blit(length_text, (10, 90))
        
        pygame.display.flip()
        self.clock.tick(30)  # Ограничиваем FPS
        
    def close(self):
        pygame.quit()
