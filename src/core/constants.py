"""Game constants and configuration values."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
POLICIES_DIR = PROJECT_ROOT / "policies"
MUJOCO_MODEL_PATH = ASSETS_DIR / "pong.xml"

# Physics
PHYSICS_TIMESTEP = 0.002  # MuJoCo timestep in seconds
SUBSTEPS = 4  # Physics substeps per game frame

# Game dimensions (must match pong.xml)
TABLE_WIDTH = 2.2  # Total width (-1.1 to 1.1)
TABLE_HEIGHT = 1.7  # Total height (-0.85 to 0.85)
BALL_RADIUS = 0.03
PADDLE_HALF_HEIGHT = 0.12
PADDLE_X_LEFT = -0.95
PADDLE_X_RIGHT = 0.95
WALL_Y = 0.8  # Ball resets if beyond this

# Scoring boundaries
SCORE_X_LEFT = -1.1  # Ball past left paddle = right scores
SCORE_X_RIGHT = 1.1  # Ball past right paddle = left scores

# Game settings
DEFAULT_BALL_SPEED = 1.5
DEFAULT_PADDLE_SENSITIVITY = 2.5
BALL_INITIAL_SPEED = 1.5  # Initial ball velocity magnitude
POINT_COUNTDOWN_SECONDS = 2.0  # Pause duration between points
MAX_SCORE = 11  # Points to win a game

# Frame rates
GAME_FPS = 60
PHYSICS_HZ = int(1.0 / PHYSICS_TIMESTEP)

# Visualization
VISER_PORT = 8080
KEYBOARD_WS_PORT = 8081

# Network visualizer
NETWORK_VIZ_UPDATE_INTERVAL = 5  # Update every N frames
MAX_NODES_PER_LAYER = 16  # Limit displayed nodes for performance

# Training defaults
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_CLIP_RANGE = 0.2
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_ENVS = 4
DEFAULT_TRAINING_TIMESTEPS = 100_000

# Reward shaping
REWARD_SCORE = 1.0
REWARD_CONCEDE = -1.0
REWARD_TRACKING = 0.001  # Small reward for tracking ball with paddle

# MuJoCo joint/actuator indices
BALL_X_JOINT = 0
BALL_Y_JOINT = 1
PADDLE_LEFT_JOINT = 2
PADDLE_RIGHT_JOINT = 3

PADDLE_LEFT_ACTUATOR = 0
PADDLE_RIGHT_ACTUATOR = 1

# Sensor indices
SENSOR_BALL_X = 0
SENSOR_BALL_Y = 1
SENSOR_BALL_VX = 2
SENSOR_BALL_VY = 3
SENSOR_PADDLE_LEFT = 4
SENSOR_PADDLE_RIGHT = 5
