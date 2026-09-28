import os
import events as e
from .callbacks import state_to_features, MODEL_FILE

ALGORITHM = "q_learning"  # Options: 'q_learning' or 'sarsa'

def setup_training(self):
    """Initializes training variables."""
    self.algorithm = ALGORITHM
    # Buffer to hold previous step data for SARSA updates
    self.last_transition = None

def reward_from_events(self, events: list) -> float:
    """
    Basic reward assignment.
    Your teammates will expand this via reward shaping.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.INVALID_ACTION: -0.1,
        e.WAITED: -0.05
    }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    """Invoked step-by-step during an episode."""
    if old_game_state is None:
        return

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    if self.algorithm == "q_learning":
        self.model.update_q_learning(old_features, self_action, reward, new_features)
    
    elif self.algorithm == "sarsa":
        if self.last_transition is not None:
            # sarsa updates the PREVIOUS transition (s, a, r) using CURRENT (s', a')
            prev_s, prev_a, prev_r = self.last_transition
            self.model.update_sarsa(prev_s, prev_a, prev_r, old_features, self_action)
        self.last_transition = (old_features, self_action, reward)

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """Terminal state update at round end and weight checkpointing."""
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)

    if self.algorithm == "q_learning":
        self.model.update_q_learning(last_features, last_action, reward, None)
    
    elif self.algorithm == "sarsa":
        if self.last_transition is not None:
            prev_s, prev_a, prev_r = self.last_transition
            self.model.update_sarsa(prev_s, prev_a, prev_r, last_features, last_action)
        # Terminal step
        self.model.update_sarsa(last_features, last_action, reward, None, None)
        self.last_transition = None

    # Save weights after each round
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    self.model.save(model_path)