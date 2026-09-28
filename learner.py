import os
import pickle
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INDEX = {a: i for i, a in enumerate(ACTIONS)}

class LinearRLModel:
    """
    Linear function approximator for Q(s, a) = w_a^T * phi(s).
    Supports both standard Q-learning and SARSA updates.
    """
    def __init__(self, feature_dim: int, alpha: float = 0.005, gamma: float = 0.95, epsilon: float = 0.2):
        self.feature_dim = feature_dim
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = ACTIONS
        self.n_actions = len(ACTIONS)
        
        # Shape: (6, feature_dim) -> One weight vector per action
        self.weights = np.zeros((self.n_actions, self.feature_dim), dtype=np.float32)

    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """Computes Q(s, a) for all actions."""
        return np.dot(self.weights, features)

    def choose_action(self, features: np.ndarray, train: bool = False) -> str:
        """Epsilon-greedy action selection."""
        if train and np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        q_values = self.get_q_values(features)
        return self.actions[int(np.argmax(q_values))]

    def update_q_learning(self, s_feat: np.ndarray, action: str, reward: float, s_prime_feat: np.ndarray):
        """
        Standard Off-Policy Q-learning update:
        target = r + gamma * max_a' Q(s', a')
        """
        a_idx = ACTION_TO_INDEX[action]
        current_q = np.dot(self.weights[a_idx], s_feat)

        if s_prime_feat is None:  # Terminal state
            td_target = reward
        else:
            q_next_max = np.max(self.get_q_values(s_prime_feat))
            td_target = reward + self.gamma * q_next_max

        td_error = td_target - current_q
        self.weights[a_idx] += self.alpha * td_error * s_feat

    def update_sarsa(self, s_feat: np.ndarray, action: str, reward: float, s_prime_feat: np.ndarray, next_action: str):
        """
        On-Policy SARSA update:
        target = r + gamma * Q(s', a')
        """
        a_idx = ACTION_TO_INDEX[action]
        current_q = np.dot(self.weights[a_idx], s_feat)

        if s_prime_feat is None:  # Terminal state
            td_target = reward
        else:
            next_a_idx = ACTION_TO_INDEX[next_action]
            q_next = np.dot(self.weights[next_a_idx], s_prime_feat)
            td_target = reward + self.gamma * q_next

        td_error = td_target - current_q
        self.weights[a_idx] += self.alpha * td_error * s_feat

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.weights, f)

    def load(self, filepath: str):
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                self.weights = pickle.load(f)