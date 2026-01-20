"""Gymnasium environment for Pong."""
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
import numpy as np
from typing import Optional, Tuple, Any, Dict

from ..core.physics_engine import PhysicsEngine
from ..core.constants import (
    REWARD_SCORE,
    REWARD_CONCEDE,
    BALL_INITIAL_SPEED,
    GAME_FPS,
)


class PongEnv(gym.Env):
    """Gymnasium environment for training RL agents to play Pong.

    The agent controls the right paddle against a simple tracking opponent.

    Observation space (5 dimensions) - NORMALIZED FOR AGENT'S PERSPECTIVE:
        - ball_x: Ball X position, negated so positive = opponent's side [-1.2, 1.2]
        - ball_y: Ball Y position [-0.9, 0.9]
        - ball_vx: Ball X velocity, negated so positive = going away [-1, 1]
        - ball_vy: Ball Y velocity [-1, 1]
        - own_paddle_y: Agent's paddle Y position [-0.65, 0.65]

    Note: Coordinates are flipped so the agent "sees" the game as if it were
    on the left side. This makes the learned policy more symmetric.

    Action space:
        - Continuous [-1, 1]: Paddle velocity (negative = down, positive = up)

    Rewards:
        - +1.0 for scoring a point
        - -1.0 for conceding a point
    """

    metadata = {"render_modes": ["human"], "render_fps": GAME_FPS}

    def __init__(
        self,
        ball_speed_multiplier: float = 1.0,
        paddle_sensitivity: float = 1.0,
        opponent_skill: float = 0.7,
        max_steps: int = 3000,
        randomize_ball_speed: bool = True,
        frame_skip: int = 12,  # ~5 Hz control at 60 FPS
        win_score: int = 5,  # Episode ends when someone reaches this score
    ):
        """Initialize the Pong environment.

        Args:
            ball_speed_multiplier: Multiplier for ball speed.
            paddle_sensitivity: Multiplier for paddle movement speed.
            opponent_skill: How well the opponent tracks the ball (0-1).
            max_steps: Maximum steps per episode.
            randomize_ball_speed: If True, randomize ball speed for domain randomization.
            frame_skip: Number of physics frames per environment step (~5 Hz at 12).
            win_score: Score needed to win (episode terminates).
        """
        super().__init__()

        self.ball_speed_multiplier = ball_speed_multiplier
        self.paddle_sensitivity = paddle_sensitivity
        self.opponent_skill = opponent_skill
        self.max_steps = max_steps
        self.randomize_ball_speed = randomize_ball_speed
        self.frame_skip = frame_skip
        self.win_score = win_score

        # Physics engine
        self.physics: Optional[PhysicsEngine] = None

        # Episode state
        self._step_count = 0
        self._score_agent = 0
        self._score_opponent = 0

        # Observation space: normalized values (5 dimensions)
        # ball_x is negated so positive = opponent side (symmetric with left paddle view)
        self.observation_space = spaces.Box(
            low=np.array([-1.2, -0.9, -1.5, -1.5, -0.65], dtype=np.float32),
            high=np.array([1.2, 0.9, 1.5, 1.5, 0.65], dtype=np.float32),
            dtype=np.float32,
        )

        # Action space: paddle velocity
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        """Get the current observation.

        The observation is transformed so the agent sees the game from a
        symmetric perspective (as if it were on the left side).

        Returns:
            Observation array.
        """
        raw_obs = self.physics.get_observation()
        # raw_obs = [ball_x, ball_y, ball_vx, ball_vy, right_paddle_y]

        # Transform to agent's perspective (flip x-axis so agent "thinks" it's on left)
        # This makes:
        # - Positive ball_x = ball on opponent's side (far from agent)
        # - Positive ball_vx = ball moving away from agent
        obs = np.array([
            -raw_obs[0],  # Negate ball_x
            raw_obs[1],   # ball_y unchanged
            -raw_obs[2] / BALL_INITIAL_SPEED,  # Negate and normalize ball_vx
            raw_obs[3] / BALL_INITIAL_SPEED,   # Normalize ball_vy
            raw_obs[4],   # paddle_y unchanged
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information.

        Returns:
            Info dictionary.
        """
        return {
            "score_agent": self._score_agent,
            "score_opponent": self._score_opponent,
            "steps": self._step_count,
        }

    def _get_opponent_action(self) -> float:
        """Simple AI opponent that tracks the ball.

        Returns:
            Opponent action [-1, 1].
        """
        # Get raw (untransformed) ball state
        ball_x, ball_y, ball_vx, ball_vy = self.physics.get_ball_state()
        paddle_y, _ = self.physics.get_paddle_positions()

        # Only track when ball is coming toward opponent (negative vx = toward left)
        if ball_vx < 0:
            target_y = ball_y
        else:
            # Return toward center when ball going away
            target_y = 0.0

        # Simple proportional control with skill-based noise
        error = target_y - paddle_y
        action = np.clip(error * 3.0, -1, 1)

        # Add noise based on inverse skill
        noise = np.random.normal(0, 0.3 * (1 - self.opponent_skill))
        action = np.clip(action + noise, -1, 1)

        return float(action * self.opponent_skill)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        # Create or reset physics
        if self.physics is None:
            self.physics = PhysicsEngine()
        else:
            self.physics.reset()

        # Reset state
        self._step_count = 0
        self._score_agent = 0
        self._score_opponent = 0

        # Randomize ball speed if enabled
        if self.randomize_ball_speed:
            self._current_ball_speed = np.random.uniform(0.8, 1.2) * self.ball_speed_multiplier
        else:
            self._current_ball_speed = self.ball_speed_multiplier

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Agent's action (paddle velocity).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._step_count += 1

        # Agent controls RIGHT paddle, opponent controls LEFT paddle
        agent_action = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # Initialize reward for this step
        reward = 0.0

        # Run multiple physics frames (frame skip for lower control frequency)
        for _ in range(self.frame_skip):
            # Get opponent action (recomputed each frame for smoother tracking)
            opponent_action = self._get_opponent_action()

            # Step physics (left=opponent, right=agent)
            left_scored, right_scored = self.physics.step(
                left_paddle_action=opponent_action,
                right_paddle_action=agent_action,
                ball_speed_mult=self._current_ball_speed,
                paddle_sensitivity=self.paddle_sensitivity,
            )

            # Handle scoring
            if right_scored:
                # Agent scored (ball went past opponent on left)
                reward += REWARD_SCORE
                self._score_agent += 1
                self.physics.reset_ball(direction=-1)  # Ball goes to opponent

            if left_scored:
                # Opponent scored (ball went past agent on right)
                reward += REWARD_CONCEDE
                self._score_opponent += 1
                self.physics.reset_ball(direction=1)  # Ball goes to agent

        # Get new observation
        new_obs = self._get_obs()

        # Check termination (someone won)
        terminated = (self._score_agent >= self.win_score or
                      self._score_opponent >= self.win_score)

        # Check truncation (max steps reached)
        truncated = self._step_count >= self.max_steps

        return new_obs, reward, terminated, truncated, self._get_info()

    def render(self):
        """Render the environment (placeholder - rendering handled by Viser)."""
        pass

    def close(self):
        """Clean up resources."""
        self.physics = None


class PongEnvForTraining(PongEnv):
    """Variant of PongEnv optimized for training.

    Uses easier opponent and optimized settings for faster learning.
    """

    def __init__(self, **kwargs):
        # Default to easier opponent for training
        kwargs.setdefault('opponent_skill', 0.5)
        kwargs.setdefault('randomize_ball_speed', True)
        kwargs.setdefault('frame_skip', 12)  # ~5 Hz control
        kwargs.setdefault('win_score', 5)  # Shorter episodes
        super().__init__(**kwargs)


def make_pong_env(
    ball_speed_multiplier: float = 1.0,
    paddle_sensitivity: float = 1.0,
    rank: int = 0,
) -> PongEnv:
    """Factory function for creating Pong environments.

    Args:
        ball_speed_multiplier: Ball speed multiplier.
        paddle_sensitivity: Paddle sensitivity.
        rank: Environment rank for seeding.

    Returns:
        Monitor-wrapped PongEnv instance.
    """
    def _init():
        env = PongEnvForTraining(
            ball_speed_multiplier=ball_speed_multiplier,
            paddle_sensitivity=paddle_sensitivity,
        )
        env.reset(seed=rank)
        # Wrap with Monitor to track episode statistics
        env = Monitor(env)
        return env

    return _init
