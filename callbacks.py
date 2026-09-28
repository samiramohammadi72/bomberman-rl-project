import os
import numpy as np
from .learner import LinearRLModel

MODEL_FILE = "model_weights.pt"
FEATURE_DIM = 10  # Define standard feature length agreed upon with your team

def setup(self):
    """Called once at the start of execution."""
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    self.model = LinearRLModel(feature_dim=FEATURE_DIM)
    
    if os.path.isfile(model_path):
        self.model.load(model_path)

def act(self, game_state: dict) -> str:
    """Invoked every step to decide an action."""
    features = state_to_features(game_state)
    return self.model.choose_action(features, train=self.train)

def state_to_features(game_state: dict) -> np.ndarray:
    """
    Placeholder: Hand this signature over to Faiq & teammate.
    Must always return a 1D float32 numpy array of shape (FEATURE_DIM,).
    """
    if game_state is None:
        return None
    # Dummy mock features for interface verification
    return np.zeros(FEATURE_DIM, dtype=np.float32)