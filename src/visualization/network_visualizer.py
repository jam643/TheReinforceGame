"""Neural network activation visualizer."""

import numpy as np
import torch
import viser
from typing import Optional, List, Dict, Tuple
import logging

from stable_baselines3 import PPO

from ..core.constants import NETWORK_VIZ_UPDATE_INTERVAL

logger = logging.getLogger(__name__)

# Semantic labels for observation inputs
INPUT_LABELS = [
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "left_paddle_y",
    "right_paddle_y",
]

OUTPUT_LABELS = [
    "paddle_vel",
]


def activation_to_color(value: float) -> Tuple[int, int, int]:
    """Convert activation value to RGB color.

    Uses a blue -> green -> red gradient.

    Args:
        value: Activation value (typically 0-1 after ReLU).

    Returns:
        RGB tuple.
    """
    # Clamp and normalize
    v = np.clip(value, -1, 2)
    v = (v + 1) / 3.0  # Map [-1, 2] to [0, 1]

    if v < 0.5:
        # Blue to green
        t = v * 2
        r = 0
        g = int(255 * t)
        b = int(255 * (1 - t))
    else:
        # Green to red
        t = (v - 0.5) * 2
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 0

    return (r, g, b)


class NetworkVisualizer:
    """Visualizes neural network activations in 3D."""

    def __init__(
        self,
        server: viser.ViserServer,
        offset: Tuple[float, float, float] = (1.5, -0.4, 0.3),
    ):
        """Initialize the network visualizer.

        Args:
            server: Viser server.
            offset: 3D offset from origin for the visualization.
                    X: left-right (negative = left of table)
                    Y: layer progression (input to output)
                    Z: vertical node stacking
        """
        self.server = server
        self.offset = offset
        self._model: Optional[PPO] = None
        self._layer_info: List[Dict] = []
        self._node_handles: Dict[str, viser.MeshHandle] = {}
        self._label_handles: List = []
        self._activations: Dict[str, np.ndarray] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._frame_count = 0

    def set_model(self, model: Optional[PPO]):
        """Set the model to visualize.

        Args:
            model: PPO model or None to clear.
        """
        # Clean up old visualization
        self._clear()

        if model is None:
            self._model = None
            return

        self._model = model
        self._extract_layers()
        self._create_visualization()
        self._register_hooks()

    def _clear(self):
        """Clear all visualization elements."""
        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Remove node meshes
        for handle in self._node_handles.values():
            try:
                handle.remove()
            except Exception:
                pass
        self._node_handles.clear()

        # Remove labels
        for handle in self._label_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._label_handles.clear()

        self._layer_info.clear()
        self._activations.clear()

        # Try to remove entire network group
        try:
            self.server.scene.remove_by_name("/network")
        except Exception:
            pass

    def _extract_layers(self):
        """Extract layer information from the model."""
        if self._model is None:
            return

        policy = self._model.policy
        mlp = policy.mlp_extractor

        # Get policy network layers
        self._layer_info = []

        # Input layer (observation size)
        obs_dim = self._model.observation_space.shape[0]
        self._layer_info.append(
            {
                "name": "input",
                "size": obs_dim,
                "module": None,
                "labels": INPUT_LABELS[:obs_dim],
            }
        )

        # Hidden layers from policy network
        if hasattr(mlp, "policy_net"):
            hidden_idx = 0
            for i, layer in enumerate(mlp.policy_net):
                if isinstance(layer, torch.nn.Linear):
                    self._layer_info.append(
                        {
                            "name": f"hidden_{hidden_idx}",
                            "size": layer.out_features,
                            "module": layer,
                            "labels": None,
                        }
                    )
                    hidden_idx += 1

        # Output layer (action dimension)
        action_dim = self._model.action_space.shape[0]
        self._layer_info.append(
            {
                "name": "output",
                "size": action_dim,
                "module": None,
                "labels": OUTPUT_LABELS[:action_dim],
            }
        )

        logger.info(
            f"Extracted {len(self._layer_info)} layers: {[l['size'] for l in self._layer_info]}"
        )

    def _create_visualization(self):
        """Create the 3D node graph visualization.

        Layout: Input on left, output on right. Nodes arranged vertically.
        Positioned to the left of the game table.

        Coordinate system:
        - X: fixed (left of table)
        - Y: layer progression (input -> output, left to right from camera)
        - Z: vertical node stacking within each layer
        """
        if not self._layer_info:
            return

        num_layers = len(self._layer_info)
        layer_spacing = 0.4  # Y spacing between layers (left-to-right)
        node_spacing = 0.06  # Z spacing between nodes (vertical)
        node_radius = 0.025

        # Calculate total depth for centering title
        total_depth = layer_spacing * (num_layers - 1)

        # Create title above the network
        title = self.server.scene.add_label(
            name="/network/title",
            text="Policy Network",
            position=(
                self.offset[0],
                self.offset[1] + total_depth / 2,
                self.offset[2] + 0.4,
            ),
        )
        self._label_handles.append(title)

        for layer_idx, layer in enumerate(self._layer_info):
            # Y position decreases with layer index (input on left, output on right)
            layer_y = self.offset[1] + total_depth - layer_idx * layer_spacing
            num_nodes = layer["size"]
            total_height = (num_nodes - 1) * node_spacing

            # Create nodes - stacked vertically (along Z axis)
            for node_idx in range(num_nodes):
                node_z = self.offset[2] + node_idx * node_spacing - total_height / 2
                node_key = f"{layer_idx}_{node_idx}"

                handle = self.server.scene.add_icosphere(
                    name=f"/network/layer_{layer_idx}/node_{node_idx}",
                    radius=node_radius,
                    position=(self.offset[0], layer_y, node_z),
                    color=(100, 100, 100),
                )
                self._node_handles[node_key] = handle

            # Add semantic labels for input layer (to the left of nodes)
            if layer_idx == 0 and layer["labels"]:
                for node_idx, label_text in enumerate(layer["labels"]):
                    if node_idx < num_nodes:
                        node_z = (
                            self.offset[2] + node_idx * node_spacing - total_height / 2
                        )
                        lbl = self.server.scene.add_label(
                            name=f"/network/layer_{layer_idx}/label_{node_idx}",
                            text=label_text,
                            position=(self.offset[0], layer_y - 0.25, node_z),
                        )
                        self._label_handles.append(lbl)

            # Add semantic labels for output layer (to the right of nodes)
            elif layer_idx == num_layers - 1 and layer["labels"]:
                for node_idx, label_text in enumerate(layer["labels"]):
                    if node_idx < num_nodes:
                        node_z = (
                            self.offset[2] + node_idx * node_spacing - total_height / 2
                        )
                        lbl = self.server.scene.add_label(
                            name=f"/network/layer_{layer_idx}/label_{node_idx}",
                            text=label_text,
                            position=(self.offset[0], layer_y + 0.15, node_z),
                        )
                        self._label_handles.append(lbl)

            # Add layer type label above each layer
            layer_label_text = layer["name"]
            lbl = self.server.scene.add_label(
                name=f"/network/layer_{layer_idx}/title",
                text=layer_label_text,
                position=(
                    self.offset[0],
                    layer_y,
                    self.offset[2] + total_height / 2 + 0.08,
                ),
            )
            self._label_handles.append(lbl)

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        if self._model is None:
            return

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._activations[name] = output.detach().cpu().numpy()

            return hook

        mlp = self._model.policy.mlp_extractor

        if hasattr(mlp, "policy_net"):
            hidden_idx = 0
            for i, layer in enumerate(mlp.policy_net):
                if isinstance(layer, torch.nn.Linear):
                    h = layer.register_forward_hook(make_hook(f"hidden_{hidden_idx}"))
                    self._hooks.append(h)
                    hidden_idx += 1

    def update(self, observation: Optional[np.ndarray] = None):
        """Update the visualization with current activations.

        Args:
            observation: Current observation to run through network.
        """
        self._frame_count += 1
        if self._frame_count % NETWORK_VIZ_UPDATE_INTERVAL != 0:
            return

        if self._model is None or observation is None:
            return

        # Run observation through network to trigger hooks
        try:
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self._model.policy.forward(
                    obs_tensor, deterministic=True
                )
                self._activations["output"] = action.detach().cpu().numpy()
        except Exception as e:
            logger.debug(f"Forward pass failed: {e}")
            return

        # Update input layer colors (based on observation values)
        for i in range(len(observation)):
            key = f"0_{i}"
            if key in self._node_handles:
                color = activation_to_color(observation[i])
                self._node_handles[key].color = color

        # Update hidden layer colors
        for layer_idx, layer in enumerate(self._layer_info[1:-1], start=1):
            name = layer["name"]
            if name in self._activations:
                activations = self._activations[name].flatten()
                for i in range(len(activations)):
                    key = f"{layer_idx}_{i}"
                    if key in self._node_handles:
                        color = activation_to_color(activations[i])
                        self._node_handles[key].color = color

        # Update output layer colors
        if "output" in self._activations:
            output_layer_idx = len(self._layer_info) - 1
            output_vals = self._activations["output"].flatten()
            for i in range(len(output_vals)):
                key = f"{output_layer_idx}_{i}"
                if key in self._node_handles:
                    color = activation_to_color(output_vals[i])
                    self._node_handles[key].color = color

    def remove(self):
        """Remove all visualization elements."""
        self._clear()
