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

        # Material Design / Game Colors
        self.BG_LIGHT = (170, 215, 81)  # Light green
        self.BG_DARK = (162, 209, 73)   # Dark green
        self.HEADER_BG = (74, 117, 44)  # Darker green header

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (231, 71, 29)        # Apple red
        self.APPLE_LEAF = (100, 200, 50) # Leaf green
        self.GHOST_ALPHA = 100          # Transparency for dead snakes
        self.GRAY = (128, 128, 128)     # Added back for fallback colors
        self.HEADER_HEIGHT = 40         # Height of the top UI bar

        # Snake Colors (Material Design paletteish)
        self.SNAKE_COLORS = [
            (66, 133, 244),   # Google Blue
            (219, 68, 55),    # Google Red
            (244, 180, 0),    # Google Yellow
            (15, 157, 88),    # Google Green
            (171, 71, 188),   # Purple
            (0, 188, 212)     # Cyan
        ]

        # Initialize game state
        self.snakes = []
        self.directions = []
        self.scores = []
        self.foods = []
        self.active_agents = set()
        self.dead_snakes = [] # Stores info about dead snakes for rendering ghosts: (snake_body, color)
        self.steps = 0
        self.game_over = False

        # Initialize Pygame
        pygame.init()
        try:
            pygame.mixer.init()
            self.mixer_initialized = True
        except pygame.error:
            print("Warning: No audio device found. Sound will be disabled.")
            self.mixer_initialized = False

        # Increase window height to accommodate header
        self.screen = pygame.display.set_mode((width, height + self.HEADER_HEIGHT))
        pygame.display.set_caption('Multi-Agent Snake AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Initialize Sounds
        self._init_sounds()

        self.reset()

    def _init_sounds(self):
        """Generates simple beep sounds using numpy."""
        if not self.mixer_initialized:
            self.sound_enabled = False
            return

        try:
            # Eat sound (high pitch beep)
            duration = 0.1
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            note = np.sin(2 * np.pi * 880 * t)  # 880 Hz
            # Ensure 16-bit signed integer format
            audio = (note * 32767).astype(np.int16)
            # Stereo
            audio = np.column_stack((audio, audio))
            self.eat_sound = pygame.sndarray.make_sound(audio)

            # Die sound (low pitch descent)
            duration = 0.3
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Frequency sweep from 440 to 100
            freqs = np.linspace(440, 100, len(t))
            note = np.sin(2 * np.pi * freqs * t)
            audio = (note * 32767).astype(np.int16)
            audio = np.column_stack((audio, audio))
            self.die_sound = pygame.sndarray.make_sound(audio)

            self.sound_enabled = True
        except Exception as e:
            print(f"Sound initialization failed: {e}")
            self.sound_enabled = False

    def _play_sound(self, sound_type):
        if not self.sound_enabled:
            return
        if sound_type == 'eat':
            self.eat_sound.play()
        elif sound_type == 'die':
            self.die_sound.play()

    def _place_food(self):
        max_attempts = 100
        attempts = 0
        while attempts < max_attempts:
            food = (random.randint(0, self.n_cells_x-1),
                   random.randint(0, self.n_cells_y-1))
            # Check food is not on any snake
            if not any(food in snake for snake in self.snakes):
                # Check food is not same as other food
                if food not in self.foods:
                    return food
            attempts += 1
        return (random.randint(0, self.n_cells_x-1),
                random.randint(0, self.n_cells_y-1))

    def reset(self):
        self.snakes = []
        self.directions = []
        self.scores = [0] * self.n_agents
        self.foods = []
        self.steps = 0
        self.active_agents = set(range(self.n_agents))
        self.dead_snakes = []
        self.game_over = False

        start_positions = []
        initial_directions = []

        if self.n_agents == 1:
            start_positions.append((self.n_cells_x // 2, self.n_cells_y // 2))
            initial_directions.append((1, 0))
        else:
            margin = 2
            usable_width = self.n_cells_x - 2 * margin
            usable_height = self.n_cells_y - 2 * margin
            perimeter = 2 * (usable_width + usable_height)

            for i in range(self.n_agents):
                pos = (i * perimeter) // self.n_agents
                if pos < usable_width:
                    x = margin + pos
                    y = margin
                    direction = (0, 1)
                elif pos < usable_width + usable_height:
                    x = margin + usable_width
                    y = margin + (pos - usable_width)
                    direction = (-1, 0)
                elif pos < 2 * usable_width + usable_height:
                    x = margin + usable_width - (pos - (usable_width + usable_height))
                    y = margin + usable_height
                    direction = (0, -1)
                else:
                    x = margin
                    y = margin + usable_height - (pos - (2 * usable_width + usable_height))
                    direction = (1, 0)

                start_positions.append((int(x), int(y)))
                initial_directions.append(direction)

        for i in range(self.n_agents):
            self.snakes.append([start_positions[i]])
            self.directions.append(initial_directions[i])

        food = self._place_food()
        if food is not None:
            self.foods.append(food)

        return [self._get_state(i) for i in range(self.n_agents)]

    def _get_state(self, agent_idx):
        if agent_idx not in self.active_agents:
            return np.zeros(17)

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

        closest_food = min(self.foods, key=lambda f: abs(f[0] - head[0]) + abs(f[1] - head[1]))

        state = [
            (dir_r and self._is_collision(point_r, agent_idx)) or 
            (dir_l and self._is_collision(point_l, agent_idx)) or 
            (dir_u and self._is_collision(point_u, agent_idx)) or 
            (dir_d and self._is_collision(point_d, agent_idx)),

            (dir_u and self._is_collision(point_r, agent_idx)) or 
            (dir_d and self._is_collision(point_l, agent_idx)) or 
            (dir_l and self._is_collision(point_u, agent_idx)) or 
            (dir_r and self._is_collision(point_d, agent_idx)),

            (dir_d and self._is_collision(point_r, agent_idx)) or 
            (dir_u and self._is_collision(point_l, agent_idx)) or 
            (dir_r and self._is_collision(point_u, agent_idx)) or 
            (dir_l and self._is_collision(point_d, agent_idx)),

            dir_l, dir_r, dir_u, dir_d,

            closest_food[0] < head[0],
            closest_food[0] > head[0],
            closest_food[1] < head[1],
            closest_food[1] > head[1],

            len(self.snakes[agent_idx]) / (self.n_cells_x * self.n_cells_y),

            self._get_nearest_snake_distance(agent_idx) / (self.n_cells_x + self.n_cells_y),
            self._get_nearest_snake_direction_x(agent_idx),
            self._get_nearest_snake_direction_y(agent_idx),
            len(self.active_agents) / self.n_agents,
            self.scores[agent_idx] / max(max(self.scores), 1)
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
        if point[0] >= self.n_cells_x or point[0] < 0 or \
           point[1] >= self.n_cells_y or point[1] < 0:
            return True

        for i, snake in enumerate(self.snakes):
            if i == agent_idx:
                if point in snake[1:]:
                    return True
            else:
                if point in snake:
                    return True
        return False

    def step(self, actions):
        self.steps += 1
        rewards = [0] * self.n_agents

        for agent_idx in list(self.active_agents):
            action = actions[agent_idx]
            if action == 1:  # Right
                self.directions[agent_idx] = (self.directions[agent_idx][1], -self.directions[agent_idx][0])
            elif action == 2:  # Left
                self.directions[agent_idx] = (-self.directions[agent_idx][1], self.directions[agent_idx][0])

        prev_distances = []
        for agent_idx in list(self.active_agents):
            head = self.snakes[agent_idx][0]
            food = min(self.foods, key=lambda f: abs(f[0] - head[0]) + abs(f[1] - head[1]))
            prev_distances.append(abs(food[0] - head[0]) + abs(food[1] - head[1]))

        for i, agent_idx in enumerate(list(self.active_agents)):
            head = self.snakes[agent_idx][0]
            direction = self.directions[agent_idx]
            new_head = (head[0] + direction[0], head[1] + direction[1])

            if self._is_collision(new_head, agent_idx):
                self._play_sound('die')
                # Add snake to ghosts before removing
                color = self.SNAKE_COLORS[agent_idx % len(self.SNAKE_COLORS)]
                self.dead_snakes.append({'body': list(self.snakes[agent_idx]), 'color': color})

                self.active_agents.remove(agent_idx)
                rewards[agent_idx] = -10
                continue

            self.snakes[agent_idx].insert(0, new_head)

            food_eaten = False
            for i, food in enumerate(self.foods):
                if new_head == food:
                    self._play_sound('eat')
                    self.scores[agent_idx] += 1
                    rewards[agent_idx] = 10
                    self.foods[i] = self._place_food()
                    food_eaten = True
                    break

            if not food_eaten:
                self.snakes[agent_idx].pop()
                curr_food = min(self.foods, key=lambda f: abs(f[0] - new_head[0]) + abs(f[1] - new_head[1]))
                curr_distance = abs(curr_food[0] - new_head[0]) + abs(curr_food[1] - new_head[1])

                if curr_distance < prev_distances[i]:
                    rewards[agent_idx] = 0.1
                else:
                    rewards[agent_idx] = -0.1

                if len(self.snakes[agent_idx]) > 4:
                    last_positions = self.snakes[agent_idx][:4]
                    unique_positions = set(last_positions)
                    if len(unique_positions) < 3:
                        rewards[agent_idx] -= 0.5

                if new_head not in self.snakes[agent_idx][1:]:
                    rewards[agent_idx] += 0.05

                rewards[agent_idx] += 0.01

        if self.n_agents == 1:
            self.game_over = len(self.active_agents) == 0 or self.steps >= 1000
        else:
            self.game_over = len(self.active_agents) <= 1 or self.steps >= 1000

        if self.game_over:
            if self.n_agents > 1 and len(self.active_agents) == 1:
                last_agent = list(self.active_agents)[0]
                rewards[last_agent] += 20
            elif self.steps >= 1000:
                for agent_idx in self.active_agents:
                    rewards[agent_idx] -= 5

        return [self._get_state(i) for i in range(self.n_agents)], rewards, self.game_over

    def _draw_apple(self, x, y):
        # Add header offset
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2 + self.HEADER_HEIGHT
        radius = self.cell_size // 2 - 2

        # Apple body
        pygame.draw.circle(self.screen, self.RED, (center_x, center_y), radius)

        # Leaf
        leaf_start = (center_x, center_y - radius)
        leaf_end = (center_x + radius // 2, center_y - radius - 4)
        pygame.draw.line(self.screen, self.APPLE_LEAF, leaf_start, leaf_end, 3)

    def _draw_snake_segment(self, x, y, color, is_head=False, direction=None, is_ghost=False):
        # Add header offset to y coordinate
        screen_y = y * self.cell_size + self.HEADER_HEIGHT

        if is_ghost:
            # For ghost, we need a surface with alpha
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            # Add alpha to color
            ghost_color = (*color, self.GHOST_ALPHA)
            pygame.draw.circle(s, ghost_color, (self.cell_size//2, self.cell_size//2), self.cell_size//2)
            self.screen.blit(s, (x * self.cell_size, screen_y))
        else:
            # Draw rounded segment (circle)
            center = (x * self.cell_size + self.cell_size // 2,
                      screen_y + self.cell_size // 2)
            pygame.draw.circle(self.screen, color, center, self.cell_size // 2)

        if is_head and direction is not None:
            # Eyes
            eye_radius = self.cell_size // 6
            eye_offset_x = self.cell_size // 4
            eye_offset_y = self.cell_size // 4

            center_x = x * self.cell_size + self.cell_size // 2
            center_y = screen_y + self.cell_size // 2

            dx, dy = direction

            if dx == 1: # Right
                eye1_pos = (center_x + eye_offset_x, center_y - eye_offset_y)
                eye2_pos = (center_x + eye_offset_x, center_y + eye_offset_y)
            elif dx == -1: # Left
                eye1_pos = (center_x - eye_offset_x, center_y - eye_offset_y)
                eye2_pos = (center_x - eye_offset_x, center_y + eye_offset_y)
            elif dy == 1: # Down
                eye1_pos = (center_x - eye_offset_x, center_y + eye_offset_y)
                eye2_pos = (center_x + eye_offset_x, center_y + eye_offset_y)
            else: # Up
                eye1_pos = (center_x - eye_offset_x, center_y - eye_offset_y)
                eye2_pos = (center_x + eye_offset_x, center_y - eye_offset_y)

            pygame.draw.circle(self.screen, self.WHITE, eye1_pos, eye_radius)
            pygame.draw.circle(self.screen, self.WHITE, eye2_pos, eye_radius)

            # Pupils (Black)
            pygame.draw.circle(self.screen, self.BLACK, eye1_pos, eye_radius // 2)
            pygame.draw.circle(self.screen, self.BLACK, eye2_pos, eye_radius // 2)


    def render(self):
        self.screen.fill(self.BG_LIGHT)

        # Draw checkerboard (shifted by HEADER_HEIGHT)
        for x in range(self.n_cells_x):
            for y in range(self.n_cells_y):
                if (x + y) % 2 == 0:
                    rect = pygame.Rect(x * self.cell_size,
                                     y * self.cell_size + self.HEADER_HEIGHT,
                                     self.cell_size,
                                     self.cell_size)
                    pygame.draw.rect(self.screen, self.BG_DARK, rect)

        # Draw ghosts (dead snakes)
        for ghost in self.dead_snakes:
             for segment in ghost['body']:
                self._draw_snake_segment(segment[0], segment[1], ghost['color'], is_ghost=True)

        # Draw snakes
        for agent_idx, snake in enumerate(self.snakes):
            if agent_idx not in self.active_agents:
                continue

            color = self.SNAKE_COLORS[agent_idx % len(self.SNAKE_COLORS)]
            direction = self.directions[agent_idx]

            # Draw body first
            for i, segment in enumerate(snake):
                is_head = (i == 0)
                self._draw_snake_segment(segment[0], segment[1], color, is_head, direction if is_head else None)

        # Draw food
        for food in self.foods:
            self._draw_apple(food[0], food[1])

        # Draw Header
        pygame.draw.rect(self.screen, self.HEADER_BG, (0, 0, self.width, self.HEADER_HEIGHT))

        # Draw Scores
        x_offset = 10

        # Sound Icon (Placeholder visual)
        # Draw a simple speaker cone
        speaker_color = self.WHITE if self.sound_enabled else self.GRAY
        pygame.draw.polygon(self.screen, speaker_color, [(self.width - 30, 20), (self.width - 15, 10), (self.width - 15, 30)])
        pygame.draw.rect(self.screen, speaker_color, (self.width - 35, 16, 5, 8))

        # Draw scores for each snake
        for i in range(self.n_agents):
            color = self.SNAKE_COLORS[i % len(self.SNAKE_COLORS)]
            status_alpha = 255 if i in self.active_agents else 100

            # Draw snake icon
            pygame.draw.circle(self.screen, color, (x_offset + 10, 20), 8)

            score_text = self.small_font.render(str(self.scores[i]), True, self.WHITE)
            self.screen.blit(score_text, (x_offset + 25, 12))

            x_offset += 60

        pygame.display.flip()
        self.clock.tick(30)  # FPS limit

    def close(self):
        pygame.quit()
