"""Real-time training statistics plotting."""
import viser
from typing import Optional, List
from collections import deque
import plotly.graph_objects as go


class TrainingPlotter:
    """Tracks and plots training statistics in real-time."""

    def __init__(
        self,
        max_history: int = 500,
    ):
        """Initialize the training plotter.

        Args:
            max_history: Maximum number of data points to keep.
        """
        self.max_history = max_history

        # Training data
        self._timesteps: deque = deque(maxlen=max_history)
        self._mean_rewards: deque = deque(maxlen=max_history)
        self._episode_counts: deque = deque(maxlen=max_history)
        self._episode_rewards: deque = deque(maxlen=max_history)

        # Current training state
        self._is_training = False
        self._current_timesteps = 0
        self._total_timesteps = 0
        self._current_episodes = 0
        self._current_mean_reward = 0.0

        # GUI elements
        self._status_text: Optional[viser.GuiInputHandle] = None
        self._progress_text: Optional[viser.GuiInputHandle] = None
        self._reward_plot_handle: Optional[viser.GuiPlotlyHandle] = None
        self._episodes_plot_handle: Optional[viser.GuiPlotlyHandle] = None

    def setup_gui(self, server: viser.ViserServer):
        """Set up the GUI elements for training plots.

        Args:
            server: Viser server to add GUI elements to.
        """
        # Status display
        with server.gui.add_folder("Training Status"):
            self._status_text = server.gui.add_text(
                "Status",
                initial_value="Idle",
                disabled=True,
            )
            self._progress_text = server.gui.add_text(
                "Progress",
                initial_value="0 / 0 timesteps",
                disabled=True,
            )

        # Reward plot
        with server.gui.add_folder("Training Reward"):
            fig = self._create_reward_figure()
            self._reward_plot_handle = server.gui.add_plotly(
                figure=fig,
                aspect=1.5,
            )

        # Episodes plot
        with server.gui.add_folder("Training Episodes"):
            fig = self._create_episodes_figure()
            self._episodes_plot_handle = server.gui.add_plotly(
                figure=fig,
                aspect=1.5,
            )

            # Reset button
            reset_btn = server.gui.add_button("Clear Training History")

            @reset_btn.on_click
            def _on_reset(event: viser.GuiEvent):
                self.reset()

    def _create_reward_figure(self) -> go.Figure:
        """Create the plotly figure for reward plot."""
        timesteps = list(self._timesteps) if self._timesteps else [0]
        mean_rewards = list(self._mean_rewards) if self._mean_rewards else [0]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timesteps,
            y=mean_rewards,
            mode='lines',
            name='Mean Reward (100 ep)',
            line=dict(color='#69db7c', width=2),
            fill='tozeroy',
            fillcolor='rgba(105, 219, 124, 0.2)',
        ))

        fig.update_layout(
            title=dict(text='Mean Episode Reward', font=dict(size=11)),
            xaxis=dict(title='Timesteps', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(title='Reward', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1),
            margin=dict(l=40, r=10, t=30, b=25),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=9),
            height=140,
        )

        return fig

    def _create_episodes_figure(self) -> go.Figure:
        """Create the plotly figure for episodes plot."""
        timesteps = list(self._timesteps) if self._timesteps else [0]
        episodes = list(self._episode_counts) if self._episode_counts else [0]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timesteps,
            y=episodes,
            mode='lines',
            name='Episodes',
            line=dict(color='#4dabf7', width=2),
        ))

        fig.update_layout(
            title=dict(text='Episodes Completed', font=dict(size=11)),
            xaxis=dict(title='Timesteps', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(title='Episodes', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1),
            margin=dict(l=40, r=10, t=30, b=25),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=9),
            height=120,
        )

        return fig

    def update(
        self,
        timesteps: int,
        total_timesteps: int,
        mean_reward: float,
        episodes: int,
    ):
        """Update training statistics.

        Args:
            timesteps: Current timesteps completed.
            total_timesteps: Total timesteps for training.
            mean_reward: Mean reward over last 100 episodes.
            episodes: Total episodes completed.
        """
        self._is_training = True
        self._current_timesteps = timesteps
        self._total_timesteps = total_timesteps
        self._current_episodes = episodes
        self._current_mean_reward = mean_reward

        # Store data points
        self._timesteps.append(timesteps)
        self._mean_rewards.append(mean_reward)
        self._episode_counts.append(episodes)

        # Update status text
        if self._status_text is not None:
            progress_pct = (timesteps / total_timesteps * 100) if total_timesteps > 0 else 0
            self._status_text.value = f"Training... {progress_pct:.1f}%"

        if self._progress_text is not None:
            self._progress_text.value = f"{timesteps:,} / {total_timesteps:,} steps | {episodes} episodes | reward: {mean_reward:.2f}"

        # Update plots
        if self._reward_plot_handle is not None:
            self._reward_plot_handle.figure = self._create_reward_figure()
        if self._episodes_plot_handle is not None:
            self._episodes_plot_handle.figure = self._create_episodes_figure()

    def set_status(self, status: str):
        """Set the status message.

        Args:
            status: Status message to display.
        """
        if self._status_text is not None:
            self._status_text.value = status
        self._is_training = "Training" in status or "training" in status

    def on_training_complete(self, success: bool, message: str):
        """Handle training completion.

        Args:
            success: Whether training completed successfully.
            message: Completion message.
        """
        self._is_training = False
        if self._status_text is not None:
            self._status_text.value = "Complete" if success else "Failed"
        if self._progress_text is not None:
            self._progress_text.value = message

    def reset(self):
        """Reset all training data."""
        self._timesteps.clear()
        self._mean_rewards.clear()
        self._episode_counts.clear()
        self._episode_rewards.clear()
        self._is_training = False
        self._current_timesteps = 0
        self._total_timesteps = 0
        self._current_episodes = 0
        self._current_mean_reward = 0.0

        if self._status_text is not None:
            self._status_text.value = "Idle"
        if self._progress_text is not None:
            self._progress_text.value = "0 / 0 timesteps"

        # Update plots
        if self._reward_plot_handle is not None:
            self._reward_plot_handle.figure = self._create_reward_figure()
        if self._episodes_plot_handle is not None:
            self._episodes_plot_handle.figure = self._create_episodes_figure()

    @property
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training
