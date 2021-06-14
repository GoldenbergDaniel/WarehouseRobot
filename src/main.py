import numpy as np

EPSILON = 0.9
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.9


class Agent:
    def __init__(self):
        self.row = 0
        self.col = 0


class Environment:
    def __init__(self):
        self.rows = 11
        self.cols = 11
        self.q_values = np.zeros((self.rows, self.cols, 4))
        self.actions = ["up", "down", "right", "left"]
        self.rewards = np.full((self.rows, self.cols), -100)
    def set_target_location(self, row, col):
        self.rewards[row, col] = 100
    def set_wall_locations(self, locations):
        for row in range(1, 10):
            for col in locations[row]:
                self.rewards[row][col] = -1
    def is_terminal_state(self, row, col):
        return not self.rewards[row][col] == -1
    def get_starting_location(self):
        row = np.random.randint(self.rows)
        col = np.random.randint(self.cols)
        while self.is_terminal_state(row, col):
            row = np.random.randint(self.rows)
            col = np.random.randint(self.cols)
        return row, col
    def get_next_action(self, row, col, epsilon):
        if np.random.random() < epsilon:
            return np.argmax(self.q_values[row, col])
        else:
            return np.random.randint(4)
    def get_next_location(self, row, col, action):
        new_row = row
        new_col = col
        if self.actions[action] == "up" and row > 0:
            new_row -= 1
        elif self.actions[action] == "right" and col < self.cols - 1:
            new_col += 1
        elif self.actions[action] == "down" and row < self.rows - 1:
            new_row += 1
        elif self.actions[action] == "left" and col > 0:
            new_col -= 1
        return new_row, new_col
    def get_shortest_path(self, row, col):
        if self.is_terminal_state(row, col):
            return []
        else:
            current_row, current_col = row, col
            shortest_path = []
            shortest_path.append([current_row, current_col])
            while not self.is_terminal_state(current_row, current_col):
                action_index = self.get_next_action(current_row, current_col, 1)
                current_row, current_col = self.get_next_location(current_row, current_col, action_index)
                shortest_path.append([current_row, current_col])
            return shortest_path
    def learn_shortest_path(self, agent, iterations):
        for _ in range(iterations):
            agent.row, agent.col = self.get_starting_location()
            while not self.is_terminal_state(agent.row, agent.col):
                # choose action
                action = self.get_next_action(agent.row, agent.col, EPSILON)

                # perform action and move to next state
                old_row = agent.row
                old_col = agent.col
                agent.row, agent.col = self.get_next_location(agent.row, agent.col, action)

                # recieve reward and calculate TD
                reward = self.rewards[agent.row, agent.col]
                old_q_value = self.q_values[old_row, old_col, action]
                temporal_difference = reward + (DISCOUNT_FACTOR * np.max(self.q_values[agent.row, agent.col])) - old_q_value

                # update Q-value for previous state
                new_q_value = old_q_value + (LEARNING_RATE * temporal_difference)
                self.q_values[old_row, old_col, action] = new_q_value


robot = Agent()
warehouse = Environment()

aisles = {}
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

warehouse.set_target_location(0, 5)
warehouse.set_wall_locations(aisles)
warehouse.learn_shortest_path(robot, 1000)

print("Warehouse:")
print(warehouse.rewards)
print("Shortest path: ", warehouse.get_shortest_path(1, 3))
