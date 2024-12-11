import pygame
import numpy as np
import random

class SnakeGame:
    def __init__(self, width=640, height=480, cell_size=20, n_agents=4):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.n_cells_x = width // cell_size
        self.n_cells_y = height // cell_size
        self.n_agents = n_agents
        
        # Цвета
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        # Цвета для разных змеек
        self.SNAKE_COLORS = [
            (0, 255, 0),    # Зеленый
            (0, 0, 255),    # Синий
            (255, 165, 0),  # Оранжевый
            (255, 0, 255),  # Пурпурный
            (0, 255, 255),  # Голубой
            (255, 255, 0)   # Желтый
        ]
        
        # Инициализация игровых переменных
        self.snakes = []
        self.directions = []
        self.scores = []
        self.foods = []
        self.active_agents = set()
        self.steps = 0
        self.game_over = False
        
        # Инициализация Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Multi-Agent Snake AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.reset()
        
    def _place_food(self):
        max_attempts = 100
        attempts = 0
        while attempts < max_attempts:
            food = (random.randint(0, self.n_cells_x-1),
                   random.randint(0, self.n_cells_y-1))
            # Проверяем, что еда не находится на змейках
            if not any(food in snake for snake in self.snakes):
                # Проверяем, что еда не совпадает с другой едой
                if food not in self.foods:
                    return food
            attempts += 1
        # Если не удалось найти место для еды, пробуем разместить в случайной точке
        return (random.randint(0, self.n_cells_x-1),
                random.randint(0, self.n_cells_y-1))
                
    def reset(self):
        # Очищаем списки
        self.snakes = []
        self.directions = []
        self.scores = [0] * self.n_agents
        self.foods = []
        self.steps = 0
        self.active_agents = set(range(self.n_agents))
        self.game_over = False
        
        # Генерируем начальные позиции для змеек
        start_positions = []
        initial_directions = []
        
        if self.n_agents == 1:
            # Для одной змейки - начало в центре
            start_positions.append((self.n_cells_x // 2, self.n_cells_y // 2))
            initial_directions.append((1, 0))  # Движение вправо
        else:
            # Для нескольких змеек - распределение по периметру
            margin = 2
            usable_width = self.n_cells_x - 2 * margin
            usable_height = self.n_cells_y - 2 * margin
            perimeter = 2 * (usable_width + usable_height)
            
            # Распределяем змеек равномерно по периметру
            for i in range(self.n_agents):
                # Определяем позицию на периметре (0 до perimeter)
                pos = (i * perimeter) // self.n_agents
                
                # Определяем, на какой стороне периме��ра находится змейка
                if pos < usable_width:  # Верхняя сторона
                    x = margin + pos
                    y = margin
                    direction = (0, 1)  # Вниз
                elif pos < usable_width + usable_height:  # Правая сторона
                    x = margin + usable_width
                    y = margin + (pos - usable_width)
                    direction = (-1, 0)  # Влево
                elif pos < 2 * usable_width + usable_height:  # Нижняя сторона
                    x = margin + usable_width - (pos - (usable_width + usable_height))
                    y = margin + usable_height
                    direction = (0, -1)  # Вверх
                else:  # Левая сторона
                    x = margin
                    y = margin + usable_height - (pos - (2 * usable_width + usable_height))
                    direction = (1, 0)  # Вправо
                
                start_positions.append((int(x), int(y)))
                initial_directions.append(direction)
        
        for i in range(self.n_agents):
            self.snakes.append([start_positions[i]])
            self.directions.append(initial_directions[i])
        
        # Размещаем еду (одну на двоих)
        food = self._place_food()
        if food is not None:
            self.foods.append(food)
        
        return [self._get_state(i) for i in range(self.n_agents)]
        
    def _get_state(self, agent_idx):
        if agent_idx not in self.active_agents:
            return np.zeros(17)  # Расширенное состояние для неактивного агента
            
        head = self.snakes[agent_idx][0]
        direction = self.directions[agent_idx]
        
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)
        
        dir_l = direction == (-1, 0)
        dir_r = direction == (1, 0)
        dir_u = direction == (0, -1)
        dir_d = direction == (0, 1)
        
        # Находим ближайшую еду
        closest_food = min(self.foods, key=lambda f: abs(f[0] - head[0]) + abs(f[1] - head[1]))
        
        state = [
            # Опасность прямо
            (dir_r and self._is_collision(point_r, agent_idx)) or 
            (dir_l and self._is_collision(point_l, agent_idx)) or 
            (dir_u and self._is_collision(point_u, agent_idx)) or 
            (dir_d and self._is_collision(point_d, agent_idx)),
            
            # Опасность справа
            (dir_u and self._is_collision(point_r, agent_idx)) or 
            (dir_d and self._is_collision(point_l, agent_idx)) or 
            (dir_l and self._is_collision(point_u, agent_idx)) or 
            (dir_r and self._is_collision(point_d, agent_idx)),
            
            # Опасность слева
            (dir_d and self._is_collision(point_r, agent_idx)) or 
            (dir_u and self._is_collision(point_l, agent_idx)) or 
            (dir_r and self._is_collision(point_u, agent_idx)) or 
            (dir_l and self._is_collision(point_d, agent_idx)),
            
            # Направление движения
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Положение ближайшей еды
            closest_food[0] < head[0],  # еда слева
            closest_food[0] > head[0],  # еда справа
            closest_food[1] < head[1],  # еда сверху
            closest_food[1] > head[1],  # еда снизу
            
            # Длина змейки
            len(self.snakes[agent_idx]) / (self.n_cells_x * self.n_cells_y),
            
            # Дополнительная информация о других змейках
            self._get_nearest_snake_distance(agent_idx) / (self.n_cells_x + self.n_cells_y),
            self._get_nearest_snake_direction_x(agent_idx),
            self._get_nearest_snake_direction_y(agent_idx),
            len(self.active_agents) / self.n_agents,  # Доля активных агентов
            self.scores[agent_idx] / max(max(self.scores), 1)  # Нормализованный счет
        ]
        
        return np.array(state, dtype=float)
        
    def _get_nearest_snake_distance(self, agent_idx):
        head = self.snakes[agent_idx][0]
        min_distance = float('inf')
        
        for i, snake in enumerate(self.snakes):
            if i != agent_idx and i in self.active_agents:
                other_head = snake[0]
                distance = abs(other_head[0] - head[0]) + abs(other_head[1] - head[1])
                min_distance = min(min_distance, distance)
                
        return min_distance if min_distance != float('inf') else 0
        
    def _get_nearest_snake_direction_x(self, agent_idx):
        head = self.snakes[agent_idx][0]
        min_distance = float('inf')
        direction = 0
        
        for i, snake in enumerate(self.snakes):
            if i != agent_idx and i in self.active_agents:
                other_head = snake[0]
                distance = abs(other_head[0] - head[0]) + abs(other_head[1] - head[1])
                if distance < min_distance:
                    min_distance = distance
                    direction = 1 if other_head[0] > head[0] else -1 if other_head[0] < head[0] else 0
                    
        return direction
        
    def _get_nearest_snake_direction_y(self, agent_idx):
        head = self.snakes[agent_idx][0]
        min_distance = float('inf')
        direction = 0
        
        for i, snake in enumerate(self.snakes):
            if i != agent_idx and i in self.active_agents:
                other_head = snake[0]
                distance = abs(other_head[0] - head[0]) + abs(other_head[1] - head[1])
                if distance < min_distance:
                    min_distance = distance
                    direction = 1 if other_head[1] > head[1] else -1 if other_head[1] < head[1] else 0
                    
        return direction
        
    def _is_collision(self, point, agent_idx):
        # Проверяем столкновение со стенами
        if point[0] >= self.n_cells_x or point[0] < 0 or \
           point[1] >= self.n_cells_y or point[1] < 0:
            return True
            
        # Проверяем столкновение с любой змейкой (включая себя)
        for i, snake in enumerate(self.snakes):
            if i == agent_idx:
                # Для своей змейки проверяем столкновение с телом
                if point in snake[1:]:
                    return True
            else:
                # Для других змеек проверяем столкновение с головой и телом
                if point in snake:
                    return True
        return False
        
    def step(self, actions):
        self.steps += 1
        rewards = [0] * self.n_agents
        
        # Обновляем направления движения для всех активных агентов
        for agent_idx in list(self.active_agents):
            action = actions[agent_idx]
            if action == 1:  # Право
                self.directions[agent_idx] = (self.directions[agent_idx][1], -self.directions[agent_idx][0])
            elif action == 2:  # Лево
                self.directions[agent_idx] = (-self.directions[agent_idx][1], self.directions[agent_idx][0])
        
        # Сохраняем предыдущие расстояния до еды
        prev_distances = []
        for agent_idx in list(self.active_agents):
            head = self.snakes[agent_idx][0]
            food = min(self.foods, key=lambda f: abs(f[0] - head[0]) + abs(f[1] - head[1]))
            prev_distances.append(abs(food[0] - head[0]) + abs(food[1] - head[1]))
        
        # Двигаем всех змеек
        for i, agent_idx in enumerate(list(self.active_agents)):
            head = self.snakes[agent_idx][0]
            direction = self.directions[agent_idx]
            new_head = (head[0] + direction[0], head[1] + direction[1])
            
            # Проверяем столкновение
            if self._is_collision(new_head, agent_idx):
                self.active_agents.remove(agent_idx)
                rewards[agent_idx] = -10  # Штраф за столкновение
                continue
            
            self.snakes[agent_idx].insert(0, new_head)
            
            # Проверяем, съела ли змейка еду
            food_eaten = False
            for i, food in enumerate(self.foods):
                if new_head == food:
                    self.scores[agent_idx] += 1
                    rewards[agent_idx] = 10  # Награда за еду
                    self.foods[i] = self._place_food()
                    food_eaten = True
                    break
            
            if not food_eaten:
                self.snakes[agent_idx].pop()
                
                # Вычисляем награду на основе приближения к еде
                curr_food = min(self.foods, key=lambda f: abs(f[0] - new_head[0]) + abs(f[1] - new_head[1]))
                curr_distance = abs(curr_food[0] - new_head[0]) + abs(curr_food[1] - new_head[1])
                
                # Награда за приближение к еде
                if curr_distance < prev_distances[i]:
                    rewards[agent_idx] = 0.1  # Небольшая награда за приближение к еде
                else:
                    rewards[agent_idx] = -0.1  # Небольшой штраф за удаление от еды
                
                # Штраф за движение по кругу
                if len(self.snakes[agent_idx]) > 4:
                    last_positions = self.snakes[agent_idx][:4]
                    unique_positions = set(last_positions)
                    if len(unique_positions) < 3:  # Если змейка ходит по кругу
                        rewards[agent_idx] -= 0.5
                
                # Награда за исследование новой территории
                if new_head not in self.snakes[agent_idx][1:]:
                    rewards[agent_idx] += 0.05
                
                # Дополнительная награда за выживание
                rewards[agent_idx] += 0.01
        
        # Проверяем условие окончания игры
        if self.n_agents == 1:
            # Для одной змейки игра заканчивается только при столкновении или превышении лимита шагов
            self.game_over = len(self.active_agents) == 0 or self.steps >= 1000
        else:
            # Для нескольких змеек - когда осталась одна или превышен лимит шагов
            self.game_over = len(self.active_agents) <= 1 or self.steps >= 1000
        
        # Если игра не закончена, даем бонус последней выжившей змейке
        if self.game_over:
            if self.n_agents > 1 and len(self.active_agents) == 1:
                last_agent = list(self.active_agents)[0]
                rewards[last_agent] += 20  # Бонус за победу только в мультиплеере
            elif self.steps >= 1000:
                # Штраф всем за слишком долгую игру
                for agent_idx in self.active_agents:
                    rewards[agent_idx] -= 5
        
        return [self._get_state(i) for i in range(self.n_agents)], rewards, self.game_over
        
    def render(self):
        self.screen.fill(self.BLACK)
        
        # Рисуем сетку
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.width, y))
        
        # Рисуем змеек
        for agent_idx, snake in enumerate(self.snakes):
            if agent_idx not in self.active_agents:
                continue
                
            color = self.SNAKE_COLORS[agent_idx % len(self.SNAKE_COLORS)]
            for i, segment in enumerate(snake):
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
        for food in self.foods:
            food_rect = pygame.Rect(
                food[0] * self.cell_size,
                food[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, self.RED, food_rect)
        
        # Отображаем статистику
        y_offset = 10
        for i in range(self.n_agents):
            color = self.SNAKE_COLORS[i % len(self.SNAKE_COLORS)]
            status = "Active" if i in self.active_agents else "Dead"
            text = self.font.render(f'Snake {i+1}: Score {self.scores[i]} ({status})', True, color)
            self.screen.blit(text, (10, y_offset))
            y_offset += 30
            
        steps_text = self.font.render(f'Steps: {self.steps}', True, self.WHITE)
        self.screen.blit(steps_text, (10, y_offset))
        
        pygame.display.flip()
        self.clock.tick(30)  # Ограничиваем FPS
        
    def close(self):
        pygame.quit()
