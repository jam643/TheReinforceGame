"""Background RL training manager using threading."""
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Callable, List
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from ..core.constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_CLIP_RANGE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_ENVS,
    DEFAULT_TRAINING_TIMESTEPS,
    POLICIES_DIR,
)
from ..environment.pong_env import make_pong_env

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = DEFAULT_LEARNING_RATE
    discount_factor: float = DEFAULT_DISCOUNT_FACTOR
    clip_range: float = DEFAULT_CLIP_RANGE
    batch_size: int = DEFAULT_BATCH_SIZE
    n_envs: int = DEFAULT_N_ENVS
    total_timesteps: int = DEFAULT_TRAINING_TIMESTEPS
    network_arch: str = "64,64"
    ball_speed_multiplier: float = 1.0
    paddle_sensitivity: float = 1.0
    policy_name: str = "trained_policy"


class ProgressCallback(BaseCallback):
    """Callback to report training progress."""

    def __init__(
        self,
        total_timesteps: int,
        on_progress: Callable[[int, int, float, int], None],
        verbose: int = 0,
    ):
        """Initialize the callback.

        Args:
            total_timesteps: Total timesteps for training.
            on_progress: Callback function(timesteps, total, mean_reward, episodes).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.on_progress = on_progress
        self.last_report_time = 0
        self._stop_requested = False

    def request_stop(self):
        """Request training to stop."""
        self._stop_requested = True

    def _on_step(self) -> bool:
        # Check for stop signal
        if self._stop_requested:
            return False

        # Report progress periodically
        current_time = time.time()
        if current_time - self.last_report_time >= 0.5:  # Every 0.5 seconds
            self.last_report_time = current_time

            # Use SB3's internal episode info buffer
            ep_info_buffer = self.model.ep_info_buffer if hasattr(self.model, 'ep_info_buffer') else []
            if ep_info_buffer:
                mean_reward = np.mean([ep['r'] for ep in ep_info_buffer])
                episodes = len(ep_info_buffer)
            else:
                mean_reward = 0.0
                episodes = 0

            # Call the progress callback directly
            self.on_progress(
                self.num_timesteps,
                self.total_timesteps,
                float(mean_reward),
                episodes,
            )

        return True


class RLTrainer:
    """Manages background RL training."""

    def __init__(self):
        """Initialize the trainer."""
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[ProgressCallback] = None
        self._is_training = False
        self._config: Optional[TrainingConfig] = None
        self._result: Optional[dict] = None
        self._on_progress: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None

    @property
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training and self._thread is not None and self._thread.is_alive()

    def start_training(
        self,
        config: TrainingConfig,
        on_progress: Callable[[int, int, float, int], None],
        on_complete: Callable[[bool, str], None],
    ):
        """Start training in a background thread.

        Args:
            config: Training configuration.
            on_progress: Callback for progress updates (timesteps, total, mean_reward, episodes).
            on_complete: Callback when training completes (success, message).
        """
        if self.is_training:
            logger.warning("Training already in progress")
            return

        self._config = config
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._is_training = True
        self._result = None

        self._thread = threading.Thread(target=self._training_worker, daemon=True)
        self._thread.start()
        logger.info("Started training thread")

    def _training_worker(self):
        """Worker function that runs in a background thread."""
        config = self._config
        vec_env = None

        try:
            # Use CPU for MLP policies
            device = "cpu"
            logger.info(f"Training on device: {device}")

            # Parse network architecture
            try:
                hidden_layers = [int(x.strip()) for x in config.network_arch.split(',')]
            except ValueError:
                hidden_layers = [64, 64]

            # Create vectorized environment
            env_fns = [
                make_pong_env(
                    ball_speed_multiplier=config.ball_speed_multiplier,
                    paddle_sensitivity=config.paddle_sensitivity,
                    rank=i,
                )
                for i in range(config.n_envs)
            ]

            vec_env = DummyVecEnv(env_fns)

            # Create PPO model
            policy_kwargs = {
                "net_arch": dict(pi=hidden_layers, vf=hidden_layers),
            }

            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=config.learning_rate,
                gamma=config.discount_factor,
                clip_range=config.clip_range,
                batch_size=config.batch_size,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0,
            )

            # Create progress callback
            self._callback = ProgressCallback(
                total_timesteps=config.total_timesteps,
                on_progress=self._on_progress,
            )

            # Train
            logger.info(f"Starting training for {config.total_timesteps} timesteps")
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=self._callback,
                progress_bar=False,
            )

            # Save the model
            save_path = POLICIES_DIR / config.policy_name / "model.zip"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))

            # Save metadata
            import json
            from datetime import datetime
            meta = {
                "name": config.policy_name,
                "saved_at": datetime.now().isoformat(),
                "timesteps": config.total_timesteps,
                "learning_rate": config.learning_rate,
                "discount_factor": config.discount_factor,
                "network_arch": config.network_arch,
            }
            meta_path = POLICIES_DIR / config.policy_name / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            self._result = {'success': True, 'policy_name': config.policy_name}
            if self._on_complete:
                self._on_complete(True, f"Training complete! Saved: {config.policy_name}")

        except Exception as e:
            logger.exception("Training failed")
            self._result = {'success': False, 'error': str(e)}
            if self._on_complete:
                self._on_complete(False, f"Training failed: {e}")

        finally:
            if vec_env:
                vec_env.close()
            self._is_training = False
            self._callback = None

    def stop_training(self):
        """Request training to stop."""
        if not self.is_training:
            return

        if self._callback:
            self._callback.request_stop()

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=5.0)

        self._is_training = False
        logger.info("Training stopped")

    def get_result(self) -> Optional[dict]:
        """Get the training result if complete.

        Returns:
            Result dictionary or None.
        """
        return self._result

    def cleanup(self):
        """Clean up resources."""
        self.stop_training()
