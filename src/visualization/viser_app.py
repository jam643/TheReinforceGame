"""Main Viser application for the Pong RL game."""
import logging
import time
import numpy as np
import torch
import viser

from ..core.constants import (
    VISER_PORT,
    GAME_FPS,
    BALL_INITIAL_SPEED,
    POLICIES_DIR,
    POINT_COUNTDOWN_SECONDS,
)
from ..core.game_state import GameState, GameMode
from ..core.physics_engine import PhysicsEngine
from ..agents.policy_manager import PolicyManager
from ..agents.rl_trainer import RLTrainer, TrainingConfig
from .scene_renderer import SceneRenderer
from .gui_panels import GUIPanels
from .network_visualizer import NetworkVisualizer
from .reward_plotter import RewardPlotter
from .training_plotter import TrainingPlotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PongApp:
    """Main application class for the Pong RL game."""

    def __init__(self, port: int = VISER_PORT):
        """Initialize the application.

        Args:
            port: Port for the Viser server.
        """
        # Core components
        self.game_state = GameState()
        self.physics = PhysicsEngine()
        self.policy_manager = PolicyManager()
        self.trainer = RLTrainer()

        # Viser server
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        # Visualization components
        self.renderer = SceneRenderer(self.server)
        self.network_viz = NetworkVisualizer(self.server)
        self.reward_plotter = RewardPlotter()
        self.training_plotter = TrainingPlotter()

        # GUI panels
        self.gui = GUIPanels(
            server=self.server,
            game_state=self.game_state,
            on_reset=self._on_reset,
            on_pause=self._on_pause,
            on_start_training=self._on_start_training,
            on_stop_training=self._on_stop_training,
            on_load_policy=self._on_load_policy,
            on_save_policy=self._on_save_policy,
            get_policy_list=self._get_policy_list,
            reward_plotter=self.reward_plotter,
            training_plotter=self.training_plotter,
        )

        # Game state
        self._running = False
        self._paused = False
        self._score_flash_timer = 0

        # Countdown state between points
        self._countdown_time = 0.0  # Remaining countdown time in seconds
        self._countdown_direction = 1  # Direction ball will go after countdown
        self._last_countdown_display = 0  # Last displayed countdown number

        # Set up client connection handler
        @self.server.on_client_connect
        def on_connect(client: viser.ClientHandle):
            logger.info(f"Client connected: {client.client_id}")
            self.renderer.set_camera_for_client(client)

        @self.server.on_client_disconnect
        def on_disconnect(client: viser.ClientHandle):
            logger.info(f"Client disconnected: {client.client_id}")

    def _on_reset(self):
        """Handle reset button press."""
        self.physics.reset()
        self.game_state.reset_scores()
        self.gui.update_score(0, 0)
        self.reward_plotter.reset()
        logger.info("Game reset")

    def _on_pause(self):
        """Handle pause button press."""
        self._paused = not self._paused
        if self._paused:
            self.game_state.mode = GameMode.PAUSED
        else:
            self.game_state.mode = GameMode.HUMAN_VS_AI
        logger.info(f"Game {'paused' if self._paused else 'resumed'}")

    def _on_training_progress(self, timesteps: int, total: int, mean_reward: float, episodes: int):
        """Handle training progress update from callback.

        Args:
            timesteps: Current timesteps completed.
            total: Total timesteps for training.
            mean_reward: Mean reward over recent episodes.
            episodes: Total episodes completed.
        """
        progress_pct = timesteps / total if total > 0 else 0
        msg = f"Training: {timesteps:,}/{total:,} (reward: {mean_reward:.2f})"

        self.game_state.update_training_status(
            timesteps_done=timesteps,
            mean_reward=mean_reward,
            status_message=msg,
        )
        self.gui.update_training_status(msg, progress_pct)
        self.training_plotter.update(
            timesteps=timesteps,
            total_timesteps=total,
            mean_reward=mean_reward,
            episodes=episodes,
        )

    def _on_training_complete(self, success: bool, message: str):
        """Handle training completion.

        Args:
            success: Whether training completed successfully.
            message: Completion message.
        """
        self.game_state.update_training_status(
            is_training=False,
            status_message=message,
        )
        self.gui.update_training_status(message, 1.0 if success else 0.0)
        self.training_plotter.on_training_complete(success, message)

        if success:
            self.gui.update_policy_list()

        logger.info(message)

    def _on_start_training(self, timesteps: int):
        """Handle start training button.

        Args:
            timesteps: Number of timesteps to train.
        """
        if self.trainer.is_training:
            logger.warning("Training already in progress")
            return

        hyperparams = self.game_state.hyperparams
        settings = self.game_state.settings

        config = TrainingConfig(
            learning_rate=hyperparams.learning_rate,
            discount_factor=hyperparams.discount_factor,
            clip_range=hyperparams.clip_range,
            batch_size=hyperparams.batch_size,
            n_envs=hyperparams.n_envs,
            network_arch=hyperparams.network_arch,
            total_timesteps=timesteps,
            ball_speed_multiplier=settings.ball_speed_multiplier,
            paddle_sensitivity=settings.paddle_sensitivity,
            policy_name=f"trained_{int(time.time())}",
        )

        # Reset plotter before starting
        self.training_plotter.reset()
        self.training_plotter.set_status("Training started...")

        # Start training with callbacks
        self.trainer.start_training(
            config,
            on_progress=self._on_training_progress,
            on_complete=self._on_training_complete,
        )

        self.game_state.update_training_status(
            is_training=True,
            total_timesteps=timesteps,
            status_message="Training started...",
        )
        self.gui.update_training_status("Training...", 0.0)
        logger.info(f"Started training for {timesteps} timesteps")

    def _on_stop_training(self):
        """Handle stop training button."""
        self.trainer.stop_training()
        self.game_state.update_training_status(
            is_training=False,
            status_message="Training stopped",
        )
        self.gui.update_training_status("Stopped", 0.0)
        self.training_plotter.set_status("Stopped")
        logger.info("Training stopped")

    def _on_load_policy(self, name: str):
        """Handle load policy button.

        Args:
            name: Name of the policy to load.
        """
        model = self.policy_manager.load_policy(name)
        if model:
            self.game_state.current_policy = name
            self.network_viz.set_model(model)
            logger.info(f"Loaded policy: {name}")
        else:
            logger.error(f"Failed to load policy: {name}")

    def _on_save_policy(self, name: str):
        """Handle save policy button.

        Args:
            name: Name for the saved policy.
        """
        model = self.policy_manager.current_model
        if model:
            self.policy_manager.save_policy(model, name)
            self.gui.update_policy_list()
            logger.info(f"Saved policy: {name}")
        else:
            logger.warning("No policy loaded to save")

    def _get_policy_list(self) -> list:
        """Get list of available policies."""
        return self.policy_manager.get_policy_names()

    def _get_human_input(self, current_paddle_y: float) -> float:
        """Get human paddle input from mouse position.

        The paddle moves toward the mouse target Y position.

        Args:
            current_paddle_y: Current Y position of the human's paddle.

        Returns:
            Float from -1 to 1 representing paddle velocity.
        """
        target_y = self.renderer.get_mouse_target_y()
        error = target_y - current_paddle_y

        # Proportional control to move toward target
        # Gain of 5.0 gives responsive but smooth movement
        action = np.clip(error * 5.0, -1, 1)
        return float(action)

    def _get_ai_action(self, observation: np.ndarray, is_right_paddle: bool = True) -> float:
        """Get AI paddle action.

        Args:
            observation: Current observation.
            is_right_paddle: Whether this is for the right paddle.

        Returns:
            Action value from -1 to 1.
        """
        model = self.policy_manager.current_model

        if model is not None:
            try:
                # PPO expects observation as (batch, features)
                obs = observation.reshape(1, -1)
                action, _ = model.predict(obs, deterministic=True)
                return float(action[0])
            except Exception as e:
                logger.debug(f"AI action failed: {e}")

        # Fallback: simple tracking AI
        ball_y = observation[1]
        # Observation is now [ball_x, ball_y, ball_vx, ball_vy, right_paddle_y]
        # Right paddle (AI) is at index 4, left paddle needs to come from physics
        if is_right_paddle:
            paddle_y = observation[4]
        else:
            paddle_left_y, _ = self.physics.get_paddle_positions()
            paddle_y = paddle_left_y
        return np.clip((ball_y - paddle_y) * 3.0, -1, 1)

    def _update_training_status(self):
        """Check training status (callbacks handle most updates now)."""
        # Training progress is now handled by callbacks
        # This method is kept for any additional status checks if needed
        pass

    def run(self):
        """Run the main game loop."""
        self._running = True

        frame_time = 1.0 / GAME_FPS
        last_time = time.time()

        logger.info(f"Pong RL Game started!")
        logger.info(f"Open http://localhost:{VISER_PORT} in your browser")
        logger.info(f"Controls: Click on the left side of the table to move your paddle")

        try:
            while self._running:
                current_time = time.time()
                dt = current_time - last_time

                if dt < frame_time:
                    time.sleep(frame_time - dt)
                    continue

                last_time = current_time

                # Update training status
                self._update_training_status()

                # Skip physics if paused
                if self._paused or self.game_state.mode == GameMode.PAUSED:
                    continue

                # Get settings
                settings = self.game_state.settings

                # Get observation
                observation = self.physics.get_observation()
                self.game_state.last_observation = observation.tolist()

                # Get current paddle positions for human input
                paddle_left_y, paddle_right_y = self.physics.get_paddle_positions()

                # Get inputs
                if self.game_state.mode == GameMode.HUMAN_VS_AI:
                    left_action = self._get_human_input(paddle_left_y)
                    right_action = self._get_ai_action(observation, is_right_paddle=True)
                else:
                    # AI vs AI mode
                    left_action = self._get_ai_action(observation, is_right_paddle=False)
                    right_action = self._get_ai_action(observation, is_right_paddle=True)

                # Handle countdown between points
                if self._countdown_time > 0:
                    self._countdown_time -= frame_time

                    # Update countdown display
                    countdown_num = int(np.ceil(self._countdown_time))
                    if countdown_num != self._last_countdown_display:
                        self._last_countdown_display = countdown_num
                        self.renderer.show_countdown(countdown_num)

                    # During countdown, only allow paddle movement (ball stays still)
                    self.physics.step(
                        left_paddle_action=left_action,
                        right_paddle_action=right_action,
                        ball_speed_mult=0.0,  # Ball doesn't move
                        paddle_sensitivity=settings.paddle_sensitivity,
                    )

                    # When countdown ends, start the ball
                    if self._countdown_time <= 0:
                        self._countdown_time = 0
                        self.renderer.hide_countdown()
                        self.physics.reset_ball(direction=self._countdown_direction)
                else:
                    # Normal gameplay - step physics
                    left_scored, right_scored = self.physics.step(
                        left_paddle_action=left_action,
                        right_paddle_action=right_action,
                        ball_speed_mult=settings.ball_speed_multiplier,
                        paddle_sensitivity=settings.paddle_sensitivity,
                    )

                    # Handle scoring - start countdown instead of immediate ball reset
                    if left_scored:
                        self.game_state.increment_score(left=True)
                        self.renderer.flash_score(left=True)
                        self._score_flash_timer = 30
                        # Start countdown - ball will go toward the scorer (right side)
                        self._countdown_time = POINT_COUNTDOWN_SECONDS
                        self._countdown_direction = 1
                        self._last_countdown_display = 0
                        # Stop ball and center it
                        self.physics.data.qpos[0] = 0  # ball_x
                        self.physics.data.qpos[1] = 0  # ball_y
                        self.physics.data.qvel[0] = 0  # ball_vx
                        self.physics.data.qvel[1] = 0  # ball_vy

                    if right_scored:
                        self.game_state.increment_score(right=True)
                        self.renderer.flash_score(right=True)
                        self._score_flash_timer = 30
                        # Start countdown - ball will go toward the scorer (left side)
                        self._countdown_time = POINT_COUNTDOWN_SECONDS
                        self._countdown_direction = -1
                        self._last_countdown_display = 0
                        # Stop ball and center it
                        self.physics.data.qpos[0] = 0
                        self.physics.data.qpos[1] = 0
                        self.physics.data.qvel[0] = 0
                        self.physics.data.qvel[1] = 0

                    # Update reward plotter (human=left, AI=right)
                    self.reward_plotter.update(
                        human_scored=left_scored,
                        ai_scored=right_scored,
                        human_paddle_y=paddle_left_y,
                        ai_paddle_y=paddle_right_y,
                        ball_y=observation[1],
                        observation=observation,
                        ai_action=right_action,
                    )

                # Update score display
                self.gui.update_score(
                    self.game_state.score_left,
                    self.game_state.score_right,
                )

                # Reset flash effect
                if self._score_flash_timer > 0:
                    self._score_flash_timer -= 1
                    if self._score_flash_timer == 0:
                        self.renderer.reset_colors()

                # Update rendering
                positions = self.physics.get_body_positions()

                self.renderer.update(
                    ball_pos=positions['ball'],
                    paddle_left_y=paddle_left_y,
                    paddle_right_y=paddle_right_y,
                )

                # Update network visualization
                self.network_viz.update(observation)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._running = False
            self.trainer.cleanup()
            self.network_viz.remove()

    def stop(self):
        """Stop the application."""
        self._running = False


def main():
    """Entry point for the application."""
    app = PongApp()
    app.run()


if __name__ == "__main__":
    main()
