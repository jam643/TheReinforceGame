"""Background RL training manager using multiprocessing."""
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
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

    def __init__(self, progress_queue: Queue, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.progress_queue = progress_queue
        self.total_timesteps = total_timesteps
        self.last_report_time = 0
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        # Check for stop signal
        try:
            # Non-blocking check
            while not self.progress_queue.empty():
                msg = self.progress_queue.get_nowait()
                if msg.get('command') == 'stop':
                    return False
        except Exception:
            pass

        # Report progress periodically
        current_time = time.time()
        if current_time - self.last_report_time >= 1.0:  # Every second
            self.last_report_time = current_time

            # Collect episode rewards from infos
            if 'infos' in self.locals:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])

            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0

            self.progress_queue.put({
                'type': 'progress',
                'timesteps': self.num_timesteps,
                'total': self.total_timesteps,
                'mean_reward': float(mean_reward),
                'episodes': len(self.episode_rewards),
            })

        return True


def _training_worker(
    config: TrainingConfig,
    progress_queue: Queue,
    result_queue: Queue,
):
    """Worker function that runs in a separate process.

    Args:
        config: Training configuration.
        progress_queue: Queue for progress updates.
        result_queue: Queue for final results.
    """
    try:
        # Use CPU for MLP policies - PPO with small MLPs runs faster on CPU
        # See: https://github.com/DLR-RM/stable-baselines3/issues/1245
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

        # Use DummyVecEnv to avoid subprocess issues
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
        callback = ProgressCallback(progress_queue, config.total_timesteps)

        # Train
        progress_queue.put({
            'type': 'status',
            'message': 'Training started',
        })

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
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

        result_queue.put({
            'success': True,
            'policy_name': config.policy_name,
            'timesteps': config.total_timesteps,
        })

    except Exception as e:
        logger.exception("Training failed")
        result_queue.put({
            'success': False,
            'error': str(e),
        })

    finally:
        vec_env.close()


class RLTrainer:
    """Manages background RL training."""

    def __init__(self):
        """Initialize the trainer."""
        self._process: Optional[Process] = None
        self._progress_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._is_training = False
        self._config: Optional[TrainingConfig] = None

    @property
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training and self._process is not None and self._process.is_alive()

    def start_training(self, config: TrainingConfig):
        """Start training in a background process.

        Args:
            config: Training configuration.
        """
        if self.is_training:
            logger.warning("Training already in progress")
            return

        self._config = config
        self._progress_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._is_training = True

        self._process = Process(
            target=_training_worker,
            args=(config, self._progress_queue, self._result_queue),
            daemon=True,
        )
        self._process.start()
        logger.info(f"Started training process (PID: {self._process.pid})")

    def stop_training(self):
        """Request training to stop."""
        if not self.is_training:
            return

        # Send stop command
        if self._progress_queue:
            self._progress_queue.put({'command': 'stop'})

        # Wait for process to finish
        if self._process:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)

        self._is_training = False
        logger.info("Training stopped")

    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get the latest progress update.

        Returns:
            Progress dictionary or None.
        """
        if not self._progress_queue:
            return None

        latest = None
        try:
            while True:
                msg = self._progress_queue.get_nowait()
                if msg.get('type') == 'progress':
                    latest = msg
                elif msg.get('type') == 'status':
                    latest = msg
        except Exception:
            pass

        return latest

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the training result if complete.

        Returns:
            Result dictionary or None.
        """
        if not self._result_queue:
            return None

        try:
            result = self._result_queue.get_nowait()
            self._is_training = False
            return result
        except Exception:
            return None

    def cleanup(self):
        """Clean up resources."""
        self.stop_training()
        if self._progress_queue:
            self._progress_queue.close()
        if self._result_queue:
            self._result_queue.close()
