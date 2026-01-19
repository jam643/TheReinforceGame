"""3D scene rendering for Pong using Viser."""
import numpy as np
import viser
from typing import Optional, Callable
import threading

from ..core.constants import (
    BALL_RADIUS,
    PADDLE_HALF_HEIGHT,
    TABLE_WIDTH,
    TABLE_HEIGHT,
)


class SceneRenderer:
    """Manages the 3D scene rendering in Viser."""

    def __init__(self, server: viser.ViserServer):
        """Initialize the scene renderer.

        Args:
            server: The Viser server instance.
        """
        self.server = server
        self._ball_handle: Optional[viser.MeshHandle] = None
        self._paddle_left_handle: Optional[viser.MeshHandle] = None
        self._paddle_right_handle: Optional[viser.MeshHandle] = None
        self._table_handle = None

        # Mouse control state
        self._mouse_target_y: float = 0.0
        self._mouse_lock = threading.Lock()

        self._setup_scene()
        self._setup_pointer_handler()

    def _setup_scene(self):
        """Set up the static scene elements."""
        # Set up camera for top-down view
        self.server.scene.set_up_direction("+z")

        # Table surface (clickable for paddle control)
        self._table_handle = self.server.scene.add_box(
            name="/table",
            dimensions=(2.2, 1.7, 0.02),
            position=(0, 0, -0.02),
            color=(25, 25, 38),
        )

        # Center line
        self.server.scene.add_box(
            name="/centerline",
            dimensions=(0.01, 1.6, 0.01),
            position=(0, 0, 0),
            color=(100, 100, 100),
        )

        # Top wall
        self.server.scene.add_box(
            name="/wall_top",
            dimensions=(2.2, 0.1, 0.06),
            position=(0, 0.85, 0),
            color=(76, 76, 76),
        )

        # Bottom wall
        self.server.scene.add_box(
            name="/wall_bottom",
            dimensions=(2.2, 0.1, 0.06),
            position=(0, -0.85, 0),
            color=(76, 76, 76),
        )

        # Ball
        self._ball_handle = self.server.scene.add_icosphere(
            name="/ball",
            radius=BALL_RADIUS,
            position=(0, 0, 0),
            color=(255, 255, 255),
        )

        # Left paddle (human - blue)
        self._paddle_left_handle = self.server.scene.add_box(
            name="/paddle_left",
            dimensions=(0.04, 0.24, 0.06),
            position=(-0.95, 0, 0),
            color=(51, 153, 255),
        )

        # Right paddle (AI - red)
        self._paddle_right_handle = self.server.scene.add_box(
            name="/paddle_right",
            dimensions=(0.04, 0.24, 0.06),
            position=(0.95, 0, 0),
            color=(255, 76, 76),
        )

        # Goal zones (subtle indicators)
        self.server.scene.add_box(
            name="/goal_left",
            dimensions=(0.02, 1.6, 0.005),
            position=(-1.08, 0, -0.01),
            color=(51, 153, 255),
            opacity=0.3,
        )

        self.server.scene.add_box(
            name="/goal_right",
            dimensions=(0.02, 1.6, 0.005),
            position=(1.08, 0, -0.01),
            color=(255, 76, 76),
            opacity=0.3,
        )

        # Countdown label (hidden initially)
        self._countdown_label = self.server.scene.add_label(
            name="/countdown",
            text="",
            position=(0, 0, 0.2),
        )

    def _setup_pointer_handler(self):
        """Set up scene-level pointer event handling."""
        # Use scene pointer callback for click events
        @self.server.scene.on_pointer_callback_removed
        def _removed():
            pass

        @self.server.scene.on_pointer_event(event_type="click")
        def on_scene_click(event: viser.ScenePointerEvent):
            if event.ray_origin is not None and event.ray_direction is not None:
                # Calculate intersection with z=0 plane (table surface)
                origin = np.array(event.ray_origin)
                direction = np.array(event.ray_direction)

                if abs(direction[2]) > 1e-6:
                    t = -origin[2] / direction[2]
                    intersection = origin + t * direction

                    # Only respond to clicks on the left half of the table
                    if intersection[0] < 0.1:
                        target_y = np.clip(intersection[1], -0.65, 0.65)
                        with self._mouse_lock:
                            self._mouse_target_y = float(target_y)

    def get_mouse_target_y(self) -> float:
        """Get the current mouse target Y position.

        Returns:
            Target Y position for the paddle.
        """
        with self._mouse_lock:
            return self._mouse_target_y

    def update(self, ball_pos: tuple, paddle_left_y: float, paddle_right_y: float):
        """Update dynamic object positions.

        Args:
            ball_pos: (x, y, z) position of the ball.
            paddle_left_y: Y position of left paddle.
            paddle_right_y: Y position of right paddle.
        """
        if self._ball_handle is not None:
            self._ball_handle.position = (ball_pos[0], ball_pos[1], ball_pos[2] if len(ball_pos) > 2 else 0)

        if self._paddle_left_handle is not None:
            self._paddle_left_handle.position = (-0.95, paddle_left_y, 0)

        if self._paddle_right_handle is not None:
            self._paddle_right_handle.position = (0.95, paddle_right_y, 0)

    def flash_score(self, left: bool = False, right: bool = False):
        """Visual feedback when a point is scored.

        Args:
            left: If True, left player scored.
            right: If True, right player scored.
        """
        # Brief color flash on the goal that was scored
        if left and self._paddle_left_handle:
            # Left player scored - flash their paddle
            self._paddle_left_handle.color = (100, 255, 100)
        if right and self._paddle_right_handle:
            # Right player scored - flash their paddle
            self._paddle_right_handle.color = (100, 255, 100)

    def reset_colors(self):
        """Reset paddle colors to default."""
        if self._paddle_left_handle:
            self._paddle_left_handle.color = (51, 153, 255)
        if self._paddle_right_handle:
            self._paddle_right_handle.color = (255, 76, 76)

    def set_camera_for_client(self, client: viser.ClientHandle):
        """Set up the camera view for a connecting client.

        Args:
            client: The Viser client handle.
        """
        # Top-down view looking at the table
        client.camera.position = (0, 0, 2.5)
        client.camera.look_at = (0, 0, 0)
        client.camera.up = (0, 1, 0)

    def show_countdown(self, seconds: int):
        """Display countdown number.

        Args:
            seconds: Number to display (0 hides the countdown).
        """
        if self._countdown_label is not None:
            if seconds > 0:
                self._countdown_label.text = str(seconds)
            else:
                self._countdown_label.text = ""

    def hide_countdown(self):
        """Hide the countdown display."""
        if self._countdown_label is not None:
            self._countdown_label.text = ""
