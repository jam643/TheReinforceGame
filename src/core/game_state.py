"""Thread-safe game state management."""
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class GameMode(Enum):
    """Game modes."""
    HUMAN_VS_AI = auto()
    AI_VS_AI = auto()
    PAUSED = auto()


@dataclass
class TrainingStatus:
    """Training progress information."""
    is_training: bool = False
    timesteps_done: int = 0
    total_timesteps: int = 0
    mean_reward: float = 0.0
    episodes_done: int = 0
    status_message: str = "Idle"

    @property
    def progress(self) -> float:
        """Return training progress as a fraction 0-1."""
        if self.total_timesteps == 0:
            return 0.0
        return min(1.0, self.timesteps_done / self.total_timesteps)


@dataclass
class GameSettings:
    """Adjustable game parameters."""
    ball_speed_multiplier: float = 1.5
    paddle_sensitivity: float = 2.5


@dataclass
class TrainingHyperparams:
    """RL training hyperparameters."""
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    clip_range: float = 0.2
    batch_size: int = 64
    n_envs: int = 4
    network_arch: str = "8,8"  # Hidden layer sizes


class GameState:
    """Thread-safe container for all shared game state."""

    def __init__(self):
        self._lock = threading.RLock()

        # Scores
        self._score_left = 0
        self._score_right = 0

        # Game mode
        self._mode = GameMode.HUMAN_VS_AI

        # Settings
        self._settings = GameSettings()
        self._hyperparams = TrainingHyperparams()

        # Training status
        self._training_status = TrainingStatus()

        # Keyboard state
        self._keys_pressed: set[str] = set()

        # Current policy name
        self._current_policy: Optional[str] = None

        # Ball state for network viz
        self._last_observation: Optional[list[float]] = None

    @property
    def score_left(self) -> int:
        with self._lock:
            return self._score_left

    @score_left.setter
    def score_left(self, value: int):
        with self._lock:
            self._score_left = value

    @property
    def score_right(self) -> int:
        with self._lock:
            return self._score_right

    @score_right.setter
    def score_right(self, value: int):
        with self._lock:
            self._score_right = value

    @property
    def mode(self) -> GameMode:
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value: GameMode):
        with self._lock:
            self._mode = value

    @property
    def settings(self) -> GameSettings:
        with self._lock:
            return GameSettings(
                ball_speed_multiplier=self._settings.ball_speed_multiplier,
                paddle_sensitivity=self._settings.paddle_sensitivity,
            )

    def update_settings(self, **kwargs):
        """Update game settings."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._settings, key):
                    setattr(self._settings, key, value)

    @property
    def hyperparams(self) -> TrainingHyperparams:
        with self._lock:
            return TrainingHyperparams(
                learning_rate=self._hyperparams.learning_rate,
                discount_factor=self._hyperparams.discount_factor,
                clip_range=self._hyperparams.clip_range,
                batch_size=self._hyperparams.batch_size,
                n_envs=self._hyperparams.n_envs,
                network_arch=self._hyperparams.network_arch,
            )

    def update_hyperparams(self, **kwargs):
        """Update training hyperparameters."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._hyperparams, key):
                    setattr(self._hyperparams, key, value)

    @property
    def training_status(self) -> TrainingStatus:
        with self._lock:
            return TrainingStatus(
                is_training=self._training_status.is_training,
                timesteps_done=self._training_status.timesteps_done,
                total_timesteps=self._training_status.total_timesteps,
                mean_reward=self._training_status.mean_reward,
                episodes_done=self._training_status.episodes_done,
                status_message=self._training_status.status_message,
            )

    def update_training_status(self, **kwargs):
        """Update training status."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._training_status, key):
                    setattr(self._training_status, key, value)

    def key_pressed(self, key: str):
        """Mark a key as pressed."""
        with self._lock:
            self._keys_pressed.add(key.lower())

    def key_released(self, key: str):
        """Mark a key as released."""
        with self._lock:
            self._keys_pressed.discard(key.lower())

    def is_key_pressed(self, key: str) -> bool:
        """Check if a key is currently pressed."""
        with self._lock:
            return key.lower() in self._keys_pressed

    def get_pressed_keys(self) -> set[str]:
        """Get all currently pressed keys."""
        with self._lock:
            return self._keys_pressed.copy()

    @property
    def current_policy(self) -> Optional[str]:
        with self._lock:
            return self._current_policy

    @current_policy.setter
    def current_policy(self, value: Optional[str]):
        with self._lock:
            self._current_policy = value

    @property
    def last_observation(self) -> Optional[list[float]]:
        with self._lock:
            return self._last_observation.copy() if self._last_observation else None

    @last_observation.setter
    def last_observation(self, value: Optional[list[float]]):
        with self._lock:
            self._last_observation = value.copy() if value else None

    def reset_scores(self):
        """Reset both scores to zero."""
        with self._lock:
            self._score_left = 0
            self._score_right = 0

    def increment_score(self, left: bool = False, right: bool = False):
        """Increment a player's score."""
        with self._lock:
            if left:
                self._score_left += 1
            if right:
                self._score_right += 1
