"""Policy save/load management."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from stable_baselines3 import PPO

from ..core.constants import POLICIES_DIR

logger = logging.getLogger(__name__)


class PolicyManager:
    """Manages saving and loading of RL policies."""

    def __init__(self, policies_dir: Optional[Path] = None):
        """Initialize the policy manager.

        Args:
            policies_dir: Directory for storing policies.
        """
        self.policies_dir = policies_dir or POLICIES_DIR
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self._current_model: Optional[PPO] = None

    def save_policy(
        self,
        model: PPO,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a policy with metadata.

        Args:
            model: The PPO model to save.
            name: Name for the policy.
            metadata: Additional metadata to save.

        Returns:
            Path to the saved policy.
        """
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        policy_dir = self.policies_dir / safe_name
        policy_dir.mkdir(exist_ok=True)

        # Save model
        model_path = policy_dir / "model.zip"
        model.save(str(model_path))

        # Save metadata
        meta = {
            "name": name,
            "saved_at": datetime.now().isoformat(),
            "timesteps": getattr(model, 'num_timesteps', 0),
            **(metadata or {}),
        }
        meta_path = policy_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved policy '{name}' to {policy_dir}")
        return policy_dir

    def load_policy(self, name: str, device: str = "auto") -> Optional[PPO]:
        """Load a policy by name.

        Args:
            name: Name of the policy to load.
            device: Device to load the model on.

        Returns:
            Loaded PPO model, or None if not found.
        """
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        policy_dir = self.policies_dir / safe_name
        model_path = policy_dir / "model.zip"

        if not model_path.exists():
            logger.warning(f"Policy '{name}' not found at {model_path}")
            return None

        try:
            model = PPO.load(str(model_path), device=device)
            self._current_model = model
            logger.info(f"Loaded policy '{name}'")
            return model
        except Exception as e:
            logger.error(f"Failed to load policy '{name}': {e}")
            return None

    def list_policies(self) -> List[Dict[str, Any]]:
        """List all available policies.

        Returns:
            List of policy metadata dictionaries.
        """
        policies = []

        for policy_dir in self.policies_dir.iterdir():
            if not policy_dir.is_dir():
                continue

            meta_path = policy_dir / "metadata.json"
            model_path = policy_dir / "model.zip"

            if not model_path.exists():
                continue

            meta = {"name": policy_dir.name}
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta.update(json.load(f))
                except Exception:
                    pass

            policies.append(meta)

        # Sort by saved_at descending
        policies.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
        return policies

    def get_policy_names(self) -> List[str]:
        """Get list of policy names.

        Returns:
            List of policy name strings.
        """
        return [p['name'] for p in self.list_policies()]

    def delete_policy(self, name: str) -> bool:
        """Delete a policy.

        Args:
            name: Name of the policy to delete.

        Returns:
            True if deleted, False if not found.
        """
        import shutil

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        policy_dir = self.policies_dir / safe_name

        if not policy_dir.exists():
            return False

        try:
            shutil.rmtree(policy_dir)
            logger.info(f"Deleted policy '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete policy '{name}': {e}")
            return False

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a policy.

        Args:
            name: Name of the policy.

        Returns:
            Metadata dictionary, or None if not found.
        """
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        meta_path = self.policies_dir / safe_name / "metadata.json"

        if not meta_path.exists():
            return None

        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    @property
    def current_model(self) -> Optional[PPO]:
        """Get the currently loaded model."""
        return self._current_model

    @current_model.setter
    def current_model(self, model: Optional[PPO]):
        """Set the current model."""
        self._current_model = model
