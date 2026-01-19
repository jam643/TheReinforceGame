"""Real-time reward plotting for gameplay."""
import numpy as np
import viser
from typing import Optional
from collections import deque
import plotly.graph_objects as go

from ..core.constants import REWARD_SCORE, REWARD_CONCEDE, REWARD_TRACKING


class RewardPlotter:
    """Tracks and plots rewards for both players in real-time."""

    def __init__(
        self,
        max_history: int = 500,
    ):
        """Initialize the reward plotter.

        Args:
            max_history: Maximum number of data points to keep.
        """
        self.max_history = max_history

        # Reward tracking
        self._human_rewards: deque = deque(maxlen=max_history)
        self._ai_rewards: deque = deque(maxlen=max_history)
        self._timestamps: deque = deque(maxlen=max_history)

        # Cumulative rewards
        self._human_cumulative = 0.0
        self._ai_cumulative = 0.0

        # Time tracking
        self._frame_count = 0

        # GUI elements (set by setup_gui)
        self._plot_handle: Optional[viser.GuiPlotlyHandle] = None

    def setup_gui(self, server: viser.ViserServer):
        """Set up the GUI elements for the plot.

        Args:
            server: Viser server to add GUI elements to.
        """
        with server.gui.add_folder("Reward Plot"):
            # Create initial empty plot
            fig = self._create_plot_figure()
            self._plot_handle = server.gui.add_plotly(
                figure=fig,
                aspect=1.5,
            )

            # Reset button
            reset_btn = server.gui.add_button("Reset Rewards")

            @reset_btn.on_click
            def _on_reset(event: viser.GuiEvent):
                self.reset()

    def _create_plot_figure(self) -> go.Figure:
        """Create the plotly figure for the reward plot.

        Returns:
            Plotly figure.
        """
        timestamps = list(self._timestamps) if self._timestamps else [0]
        human_rewards = list(self._human_rewards) if self._human_rewards else [0]
        ai_rewards = list(self._ai_rewards) if self._ai_rewards else [0]

        fig = go.Figure()

        # Human (Player 1) rewards - blue
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=human_rewards,
            mode='lines',
            name='Human (P1)',
            line=dict(color='#4488ff', width=2),
        ))

        # AI (Player 2) rewards - red
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=ai_rewards,
            mode='lines',
            name='AI (P2)',
            line=dict(color='#ff4444', width=2),
        ))

        fig.update_layout(
            title=dict(
                text='Cumulative Reward',
                font=dict(size=12),
            ),
            xaxis=dict(
                title='Time (frames)',
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
            ),
            yaxis=dict(
                title='Reward',
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
            ),
            margin=dict(l=40, r=10, t=40, b=30),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            height=250,
        )

        return fig

    def update(
        self,
        human_scored: bool = False,
        ai_scored: bool = False,
        human_paddle_y: float = 0.0,
        ai_paddle_y: float = 0.0,
        ball_y: float = 0.0,
    ):
        """Update rewards based on game events.

        Args:
            human_scored: Whether human scored this frame.
            ai_scored: Whether AI scored this frame.
            human_paddle_y: Human paddle Y position.
            ai_paddle_y: AI paddle Y position.
            ball_y: Ball Y position.
        """
        self._frame_count += 1

        # Calculate rewards for this frame
        human_reward = 0.0
        ai_reward = 0.0

        # Scoring rewards
        if human_scored:
            human_reward += REWARD_SCORE
            ai_reward += REWARD_CONCEDE

        if ai_scored:
            ai_reward += REWARD_SCORE
            human_reward += REWARD_CONCEDE

        # Tracking rewards (small bonus for staying aligned with ball)
        human_tracking = REWARD_TRACKING * max(0, 1.0 - abs(ball_y - human_paddle_y))
        ai_tracking = REWARD_TRACKING * max(0, 1.0 - abs(ball_y - ai_paddle_y))
        human_reward += human_tracking
        ai_reward += ai_tracking

        # Update cumulative rewards
        self._human_cumulative += human_reward
        self._ai_cumulative += ai_reward

        # Store data points
        self._timestamps.append(self._frame_count)
        self._human_rewards.append(self._human_cumulative)
        self._ai_rewards.append(self._ai_cumulative)

        # Update plot every 30 frames (twice per second at 60fps)
        if self._frame_count % 30 == 0 and self._plot_handle is not None:
            fig = self._create_plot_figure()
            self._plot_handle.figure = fig

    def reset(self):
        """Reset all reward tracking."""
        self._human_rewards.clear()
        self._ai_rewards.clear()
        self._timestamps.clear()
        self._human_cumulative = 0.0
        self._ai_cumulative = 0.0
        self._frame_count = 0

        # Update plot
        if self._plot_handle is not None:
            fig = self._create_plot_figure()
            self._plot_handle.figure = fig

    @property
    def human_cumulative_reward(self) -> float:
        """Get human player's cumulative reward."""
        return self._human_cumulative

    @property
    def ai_cumulative_reward(self) -> float:
        """Get AI player's cumulative reward."""
        return self._ai_cumulative
