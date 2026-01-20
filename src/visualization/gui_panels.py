"""GUI panels for the Viser interface."""

import viser
from typing import Callable, Optional, List, TYPE_CHECKING
import logging

from ..core.game_state import GameState, GameMode
from ..core.constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_CLIP_RANGE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_ENVS,
    DEFAULT_TRAINING_TIMESTEPS,
)

if TYPE_CHECKING:
    from .reward_plotter import RewardPlotter
    from .training_plotter import TrainingPlotter

logger = logging.getLogger(__name__)


class GUIPanels:
    """Creates and manages all GUI panels."""

    def __init__(
        self,
        server: viser.ViserServer,
        game_state: GameState,
        on_reset: Callable,
        on_pause: Callable,
        on_start_training: Callable,
        on_stop_training: Callable,
        on_load_policy: Callable,
        on_save_policy: Callable,
        get_policy_list: Callable,
        reward_plotter: Optional["RewardPlotter"] = None,
        training_plotter: Optional["TrainingPlotter"] = None,
    ):
        """Initialize GUI panels.

        Args:
            server: Viser server.
            game_state: Shared game state.
            on_reset: Callback for reset button.
            on_pause: Callback for pause button.
            on_start_training: Callback for start training.
            on_stop_training: Callback for stop training.
            on_load_policy: Callback for loading a policy.
            on_save_policy: Callback for saving a policy.
            get_policy_list: Function to get available policies.
            reward_plotter: Optional reward plotter instance.
            training_plotter: Optional training plotter instance.
        """
        self.server = server
        self.game_state = game_state
        self.on_reset = on_reset
        self.on_pause = on_pause
        self.on_start_training = on_start_training
        self.on_stop_training = on_stop_training
        self.on_load_policy = on_load_policy
        self.on_save_policy = on_save_policy
        self.get_policy_list = get_policy_list
        self.reward_plotter = reward_plotter
        self.training_plotter = training_plotter

        # GUI handles
        self._score_text: Optional[viser.GuiHandle] = None
        self._status_text: Optional[viser.GuiHandle] = None
        self._progress_bar: Optional[viser.GuiHandle] = None
        self._policy_dropdown: Optional[viser.GuiHandle] = None

        self._setup_panels()

    def _setup_panels(self):
        """Set up all GUI panels with tabs."""
        # Make sidebar wider
        self.server.gui.configure_theme(control_width="large", dark_mode=True)

        # Create tab group
        tab_group = self.server.gui.add_tab_group()

        # Game tab
        with tab_group.add_tab("Game"):
            self._setup_game_panel()
            self._setup_policy_panel()
            # Add reward plotter in Game tab
            if self.reward_plotter is not None:
                self.reward_plotter.setup_gui(self.server)

        # RL Training tab
        with tab_group.add_tab("RL Training"):
            self._setup_hyperparams_panel()
            self._setup_training_panel()
            # Add training plotter in RL Training tab
            if self.training_plotter is not None:
                self.training_plotter.setup_gui(self.server)

    def _setup_game_panel(self):
        """Set up the game settings panel."""
        with self.server.gui.add_folder("Game"):
            # Score display
            self._score_text = self.server.gui.add_text(
                "Score",
                initial_value="0 - 0",
                disabled=True,
            )

            # Ball speed slider
            ball_speed = self.server.gui.add_slider(
                "Ball Speed",
                min=0.5,
                max=3.0,
                step=0.1,
                initial_value=1.5,
            )

            @ball_speed.on_update
            def _on_ball_speed(event: viser.GuiEvent):
                self.game_state.update_settings(
                    ball_speed_multiplier=event.target.value
                )

            # Paddle sensitivity slider
            paddle_sens = self.server.gui.add_slider(
                "Paddle Sensitivity",
                min=0.5,
                max=5.0,
                step=0.25,
                initial_value=2.5,
            )

            @paddle_sens.on_update
            def _on_paddle_sens(event: viser.GuiEvent):
                self.game_state.update_settings(paddle_sensitivity=event.target.value)

            # Reset button
            reset_btn = self.server.gui.add_button("Reset Game")

            @reset_btn.on_click
            def _on_reset(event: viser.GuiEvent):
                self.on_reset()

            # Pause button
            pause_btn = self.server.gui.add_button("Pause/Resume")

            @pause_btn.on_click
            def _on_pause(event: viser.GuiEvent):
                self.on_pause()

            # Mode selection
            mode_dropdown = self.server.gui.add_dropdown(
                "Mode",
                options=["Human vs AI", "AI vs AI"],
                initial_value="Human vs AI",
            )

            @mode_dropdown.on_update
            def _on_mode(event: viser.GuiEvent):
                if event.target.value == "Human vs AI":
                    self.game_state.mode = GameMode.HUMAN_VS_AI
                else:
                    self.game_state.mode = GameMode.AI_VS_AI

    def _setup_hyperparams_panel(self):
        """Set up the hyperparameters panel."""
        with self.server.gui.add_folder("RL Hyperparameters"):
            # Learning rate (log scale)
            lr_slider = self.server.gui.add_slider(
                "Learning Rate (1e-x)",
                min=2.0,
                max=5.0,
                step=0.1,
                initial_value=3.5,  # 1e-3.5 ≈ 3e-4
            )

            @lr_slider.on_update
            def _on_lr(event: viser.GuiEvent):
                lr = 10 ** (-event.target.value)
                self.game_state.update_hyperparams(learning_rate=lr)

            # Discount factor
            gamma_slider = self.server.gui.add_slider(
                "Discount Factor",
                min=0.9,
                max=0.999,
                step=0.001,
                initial_value=DEFAULT_DISCOUNT_FACTOR,
            )

            @gamma_slider.on_update
            def _on_gamma(event: viser.GuiEvent):
                self.game_state.update_hyperparams(discount_factor=event.target.value)

            # Clip range
            clip_slider = self.server.gui.add_slider(
                "PPO Clip Range",
                min=0.1,
                max=0.4,
                step=0.05,
                initial_value=DEFAULT_CLIP_RANGE,
            )

            @clip_slider.on_update
            def _on_clip(event: viser.GuiEvent):
                self.game_state.update_hyperparams(clip_range=event.target.value)

            # Batch size dropdown
            batch_dropdown = self.server.gui.add_dropdown(
                "Batch Size",
                options=["32", "64", "128", "256"],
                initial_value=str(DEFAULT_BATCH_SIZE),
            )

            @batch_dropdown.on_update
            def _on_batch(event: viser.GuiEvent):
                self.game_state.update_hyperparams(batch_size=int(event.target.value))

            # Network architecture
            arch_dropdown = self.server.gui.add_dropdown(
                "Network Architecture",
                options=["8,8", "16,16", "32,32", "64,64"],
                initial_value="8,8",
            )

            @arch_dropdown.on_update
            def _on_arch(event: viser.GuiEvent):
                self.game_state.update_hyperparams(network_arch=event.target.value)

    def _setup_training_panel(self):
        """Set up the training control panel."""
        with self.server.gui.add_folder("Training Controls"):
            # Training timesteps
            timesteps_slider = self.server.gui.add_slider(
                "Timesteps (thousands)",
                min=10,
                max=500,
                step=10,
                initial_value=100,
            )

            # Number of parallel envs
            n_envs = self.server.gui.add_slider(
                "Parallel Environments",
                min=1,
                max=8,
                step=1,
                initial_value=DEFAULT_N_ENVS,
            )

            @n_envs.on_update
            def _on_n_envs(event: viser.GuiEvent):
                self.game_state.update_hyperparams(n_envs=int(event.target.value))

            # Start training button
            start_btn = self.server.gui.add_button("Start Training")

            @start_btn.on_click
            def _on_start(event: viser.GuiEvent):
                timesteps = int(timesteps_slider.value * 1000)
                self.on_start_training(timesteps)

            # Stop training button
            stop_btn = self.server.gui.add_button("Stop Training")

            @stop_btn.on_click
            def _on_stop(event: viser.GuiEvent):
                self.on_stop_training()

    def _setup_policy_panel(self):
        """Set up the policy management panel."""
        with self.server.gui.add_folder("Policy Management"):
            # Policy dropdown
            policies = self.get_policy_list()
            if not policies:
                policies = ["(none)"]

            self._policy_dropdown = self.server.gui.add_dropdown(
                "Select Policy",
                options=policies,
                initial_value=policies[0],
            )

            # Refresh button
            refresh_btn = self.server.gui.add_button("Refresh List")

            @refresh_btn.on_click
            def _on_refresh(event: viser.GuiEvent):
                self.update_policy_list()

            # Load button
            load_btn = self.server.gui.add_button("Load Policy")

            @load_btn.on_click
            def _on_load(event: viser.GuiEvent):
                policy_name = self._policy_dropdown.value
                if policy_name and policy_name != "(none)":
                    self.on_load_policy(policy_name)

            # Policy name input
            policy_name_input = self.server.gui.add_text(
                "New Policy Name",
                initial_value="my_policy",
            )

            # Save button
            save_btn = self.server.gui.add_button("Save Current Policy")

            @save_btn.on_click
            def _on_save(event: viser.GuiEvent):
                name = policy_name_input.value
                if name:
                    self.on_save_policy(name)

    def update_score(self, left: int, right: int):
        """Update the score display.

        Args:
            left: Left player score.
            right: Right player score.
        """
        if self._score_text:
            self._score_text.value = f"{left} - {right}"

    def update_training_status(self, status: str, progress: float = 0.0):
        """Update training status display.

        Args:
            status: Status message.
            progress: Progress fraction (0-1).
        """
        if self._status_text:
            self._status_text.value = status

        if self._progress_bar:
            self._progress_bar.value = int(progress * 100)

    def update_policy_list(self):
        """Refresh the policy dropdown with available policies."""
        if self._policy_dropdown:
            policies = self.get_policy_list()
            if not policies:
                policies = ["(none)"]
            self._policy_dropdown.options = policies
            if policies:
                self._policy_dropdown.value = policies[0]
