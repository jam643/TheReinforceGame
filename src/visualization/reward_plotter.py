"""Real-time reward plotting for gameplay."""
import numpy as np
import viser
from typing import Optional, List
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.constants import REWARD_SCORE, REWARD_CONCEDE, REWARD_TRACKING

# Observation labels matching the environment (5 dimensions, no opponent paddle)
OBS_LABELS = ["ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_y"]
OBS_COLORS = ["#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#da77f2"]


class RewardPlotter:
    """Tracks and plots rewards, observations, and actions in real-time."""

    def __init__(
        self,
        max_history: int = 300,
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

        # Observation tracking (5 values - no opponent paddle)
        self._observations: List[deque] = [deque(maxlen=max_history) for _ in range(5)]

        # Action tracking (AI output)
        self._ai_actions: deque = deque(maxlen=max_history)

        # Cumulative rewards
        self._human_cumulative = 0.0
        self._ai_cumulative = 0.0

        # Time tracking
        self._frame_count = 0

        # GUI elements (set by setup_gui)
        self._reward_plot_handle: Optional[viser.GuiPlotlyHandle] = None
        self._obs_plot_handle: Optional[viser.GuiPlotlyHandle] = None
        self._action_plot_handle: Optional[viser.GuiPlotlyHandle] = None

    def setup_gui(self, server: viser.ViserServer):
        """Set up the GUI elements for all plots.

        Args:
            server: Viser server to add GUI elements to.
        """
        # Reward plot
        with server.gui.add_folder("Reward Plot"):
            fig = self._create_reward_figure()
            self._reward_plot_handle = server.gui.add_plotly(
                figure=fig,
                aspect=1.5,
            )

            reset_btn = server.gui.add_button("Reset All Plots")

            @reset_btn.on_click
            def _on_reset(event: viser.GuiEvent):
                self.reset()

        # Observation plot
        with server.gui.add_folder("AI Observations"):
            fig = self._create_observations_figure()
            self._obs_plot_handle = server.gui.add_plotly(
                figure=fig,
                aspect=1.2,
            )

        # Action plot
        with server.gui.add_folder("AI Action Output"):
            fig = self._create_action_figure()
            self._action_plot_handle = server.gui.add_plotly(
                figure=fig,
                aspect=1.5,
            )

    def _create_reward_figure(self) -> go.Figure:
        """Create the plotly figure for the reward plot."""
        timestamps = list(self._timestamps) if self._timestamps else [0]
        human_rewards = list(self._human_rewards) if self._human_rewards else [0]
        ai_rewards = list(self._ai_rewards) if self._ai_rewards else [0]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=human_rewards,
            mode='lines',
            name='Human (P1)',
            line=dict(color='#4dabf7', width=2),
        ))

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=ai_rewards,
            mode='lines',
            name='AI (P2)',
            line=dict(color='#ff6b6b', width=2),
        ))

        fig.update_layout(
            title=dict(text='Cumulative Reward', font=dict(size=11)),
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(title='Reward', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=10, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=9),
            height=140,
        )

        return fig

    def _create_observations_figure(self) -> go.Figure:
        """Create the plotly figure for observations."""
        timestamps = list(self._timestamps) if self._timestamps else [0]

        fig = go.Figure()

        for i, (label, color) in enumerate(zip(OBS_LABELS, OBS_COLORS)):
            obs_data = list(self._observations[i]) if self._observations[i] else [0]
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=obs_data,
                mode='lines',
                name=label,
                line=dict(color=color, width=1.5),
            ))

        fig.update_layout(
            title=dict(text='AI Observations', font=dict(size=11)),
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(title='Value', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.0,
                xanchor='center',
                x=0.5,
                font=dict(size=7),
                bgcolor='rgba(0,0,0,0.3)',
            ),
            margin=dict(l=40, r=10, t=45, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=9),
            height=160,
        )

        return fig

    def _create_action_figure(self) -> go.Figure:
        """Create the plotly figure for AI action output."""
        timestamps = list(self._timestamps) if self._timestamps else [0]
        actions = list(self._ai_actions) if self._ai_actions else [0]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=actions,
            mode='lines',
            name='paddle_vel',
            line=dict(color='#69db7c', width=2),
            fill='tozeroy',
            fillcolor='rgba(105, 219, 124, 0.2)',
        ))

        # Add reference lines at -1 and 1 (action bounds)
        fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig.add_hline(y=-1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.2)")

        fig.update_layout(
            title=dict(text='AI Action Output', font=dict(size=11)),
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(
                title='Velocity',
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                range=[-1.3, 1.3],
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.0,
                xanchor='right',
                x=1,
                font=dict(size=8),
            ),
            margin=dict(l=40, r=10, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=9),
            height=120,
            showlegend=True,
        )

        return fig

    def update(
        self,
        human_scored: bool = False,
        ai_scored: bool = False,
        human_paddle_y: float = 0.0,
        ai_paddle_y: float = 0.0,
        ball_y: float = 0.0,
        observation: Optional[np.ndarray] = None,
        ai_action: float = 0.0,
    ):
        """Update all tracking based on game events.

        Args:
            human_scored: Whether human scored this frame.
            ai_scored: Whether AI scored this frame.
            human_paddle_y: Human paddle Y position.
            ai_paddle_y: AI paddle Y position.
            ball_y: Ball Y position.
            observation: Full observation array (6 values).
            ai_action: AI's output action value.
        """
        self._frame_count += 1

        # Calculate rewards
        human_reward = 0.0
        ai_reward = 0.0

        if human_scored:
            human_reward += REWARD_SCORE
            ai_reward += REWARD_CONCEDE

        if ai_scored:
            ai_reward += REWARD_SCORE
            human_reward += REWARD_CONCEDE

        human_tracking = REWARD_TRACKING * max(0, 1.0 - abs(ball_y - human_paddle_y))
        ai_tracking = REWARD_TRACKING * max(0, 1.0 - abs(ball_y - ai_paddle_y))
        human_reward += human_tracking
        ai_reward += ai_tracking

        self._human_cumulative += human_reward
        self._ai_cumulative += ai_reward

        # Store data points
        self._timestamps.append(self._frame_count)
        self._human_rewards.append(self._human_cumulative)
        self._ai_rewards.append(self._ai_cumulative)

        # Store observations
        if observation is not None:
            for i in range(min(5, len(observation))):
                self._observations[i].append(float(observation[i]))
        else:
            for i in range(5):
                self._observations[i].append(0.0)

        # Store AI action
        self._ai_actions.append(float(ai_action))

        # Update plots every 30 frames (twice per second at 60fps)
        if self._frame_count % 30 == 0:
            if self._reward_plot_handle is not None:
                self._reward_plot_handle.figure = self._create_reward_figure()
            if self._obs_plot_handle is not None:
                self._obs_plot_handle.figure = self._create_observations_figure()
            if self._action_plot_handle is not None:
                self._action_plot_handle.figure = self._create_action_figure()

    def reset(self):
        """Reset all tracking."""
        self._human_rewards.clear()
        self._ai_rewards.clear()
        self._timestamps.clear()
        self._human_cumulative = 0.0
        self._ai_cumulative = 0.0
        self._frame_count = 0

        for obs_deque in self._observations:
            obs_deque.clear()
        self._ai_actions.clear()

        # Update all plots
        if self._reward_plot_handle is not None:
            self._reward_plot_handle.figure = self._create_reward_figure()
        if self._obs_plot_handle is not None:
            self._obs_plot_handle.figure = self._create_observations_figure()
        if self._action_plot_handle is not None:
            self._action_plot_handle.figure = self._create_action_figure()

    @property
    def human_cumulative_reward(self) -> float:
        """Get human player's cumulative reward."""
        return self._human_cumulative

    @property
    def ai_cumulative_reward(self) -> float:
        """Get AI player's cumulative reward."""
        return self._ai_cumulative
