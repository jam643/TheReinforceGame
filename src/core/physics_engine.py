"""MuJoCo physics engine wrapper for Pong."""
import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Tuple

from .constants import (
    MUJOCO_MODEL_PATH,
    SUBSTEPS,
    BALL_INITIAL_SPEED,
    SCORE_X_LEFT,
    SCORE_X_RIGHT,
    SENSOR_BALL_X,
    SENSOR_BALL_Y,
    SENSOR_BALL_VX,
    SENSOR_BALL_VY,
    SENSOR_PADDLE_LEFT,
    SENSOR_PADDLE_RIGHT,
    PADDLE_LEFT_ACTUATOR,
    PADDLE_RIGHT_ACTUATOR,
    BALL_X_JOINT,
    BALL_Y_JOINT,
)


class PhysicsEngine:
    """Wrapper around MuJoCo for Pong physics simulation."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the physics engine.

        Args:
            model_path: Path to the MuJoCo XML model. Defaults to pong.xml.
        """
        if model_path is None:
            model_path = MUJOCO_MODEL_PATH

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Cache body and geom IDs for rendering
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.paddle_left_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "paddle_left")
        self.paddle_right_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "paddle_right")

        # Initialize ball with random velocity
        self.reset_ball()

    def reset(self):
        """Reset the entire simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.reset_ball()

    def reset_ball(self, direction: Optional[int] = None):
        """Reset ball to center with random velocity.

        Args:
            direction: -1 for left, 1 for right, None for random.
        """
        # Reset ball position to center
        self.data.qpos[BALL_X_JOINT] = 0.0
        self.data.qpos[BALL_Y_JOINT] = 0.0

        # Random direction if not specified
        if direction is None:
            direction = np.random.choice([-1, 1])

        # Random angle between -45 and 45 degrees
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)

        # Set velocity
        speed = BALL_INITIAL_SPEED
        self.data.qvel[BALL_X_JOINT] = direction * speed * np.cos(angle)
        self.data.qvel[BALL_Y_JOINT] = speed * np.sin(angle)

    def step(self, left_paddle_action: float = 0.0, right_paddle_action: float = 0.0,
             ball_speed_mult: float = 1.0, paddle_sensitivity: float = 1.0) -> Tuple[bool, bool]:
        """Advance physics simulation by one game frame.

        Args:
            left_paddle_action: Control input for left paddle [-1, 1].
            right_paddle_action: Control input for right paddle [-1, 1].
            ball_speed_mult: Multiplier for ball speed.
            paddle_sensitivity: Multiplier for paddle movement speed.

        Returns:
            Tuple of (left_scored, right_scored) booleans.
        """
        # Apply paddle controls with sensitivity (no clipping - let sensitivity scale speed)
        self.data.ctrl[PADDLE_LEFT_ACTUATOR] = left_paddle_action * paddle_sensitivity
        self.data.ctrl[PADDLE_RIGHT_ACTUATOR] = right_paddle_action * paddle_sensitivity

        # Run physics substeps
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        # After physics step, restore ball speed to target (compensates for MuJoCo damping)
        # This gives us arcade-perfect elastic collisions
        ball_vx = self.data.qvel[BALL_X_JOINT]
        ball_vy = self.data.qvel[BALL_Y_JOINT]
        current_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        target_speed = BALL_INITIAL_SPEED * ball_speed_mult

        if current_speed > 0.01:  # Avoid division by zero
            # Immediately restore to target speed (preserves direction from collision)
            scale = target_speed / current_speed
            self.data.qvel[BALL_X_JOINT] = ball_vx * scale
            self.data.qvel[BALL_Y_JOINT] = ball_vy * scale

        # Check for scoring
        ball_x = self.data.sensordata[SENSOR_BALL_X]
        left_scored = ball_x > SCORE_X_RIGHT
        right_scored = ball_x < SCORE_X_LEFT

        return left_scored, right_scored

    def get_ball_state(self) -> Tuple[float, float, float, float]:
        """Get ball position and velocity.

        Returns:
            Tuple of (x, y, vx, vy).
        """
        return (
            float(self.data.sensordata[SENSOR_BALL_X]),
            float(self.data.sensordata[SENSOR_BALL_Y]),
            float(self.data.sensordata[SENSOR_BALL_VX]),
            float(self.data.sensordata[SENSOR_BALL_VY]),
        )

    def get_paddle_positions(self) -> Tuple[float, float]:
        """Get Y positions of both paddles.

        Returns:
            Tuple of (left_y, right_y).
        """
        return (
            float(self.data.sensordata[SENSOR_PADDLE_LEFT]),
            float(self.data.sensordata[SENSOR_PADDLE_RIGHT]),
        )

    def get_observation(self) -> np.ndarray:
        """Get the full observation vector for RL.

        Returns:
            Array of [ball_x, ball_y, ball_vx, ball_vy, paddle_left_y, paddle_right_y].
        """
        return np.array([
            self.data.sensordata[SENSOR_BALL_X],
            self.data.sensordata[SENSOR_BALL_Y],
            self.data.sensordata[SENSOR_BALL_VX],
            self.data.sensordata[SENSOR_BALL_VY],
            self.data.sensordata[SENSOR_PADDLE_LEFT],
            self.data.sensordata[SENSOR_PADDLE_RIGHT],
        ], dtype=np.float32)

    def get_body_positions(self) -> dict:
        """Get positions of all game bodies for rendering.

        Returns:
            Dict with 'ball', 'paddle_left', 'paddle_right' positions.
        """
        return {
            'ball': self.data.xpos[self.ball_body_id].copy(),
            'paddle_left': self.data.xpos[self.paddle_left_body_id].copy(),
            'paddle_right': self.data.xpos[self.paddle_right_body_id].copy(),
        }

    def set_paddle_position(self, left: Optional[float] = None, right: Optional[float] = None):
        """Directly set paddle positions (useful for reset).

        Args:
            left: Y position for left paddle.
            right: Y position for right paddle.
        """
        if left is not None:
            # Find the joint index for left paddle
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "paddle_left_y")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = np.clip(left, -0.65, 0.65)

        if right is not None:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "paddle_right_y")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = np.clip(right, -0.65, 0.65)

        # Forward kinematics to update body positions
        mujoco.mj_forward(self.model, self.data)
