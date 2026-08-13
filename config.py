"""
Config module for Bob's World 3D - 2026 Next-Gen Sci-Fi Laboratory Room Engine.
Contains 2026 Next-Gen material colors, laser grid lines, cyber-orange Bob properties,
and high-precision physics constants.
"""

# ============================================================================
# PHYSICS & ENCLOSED 3D ROOM DIMENSIONS
# ============================================================================
GRAVITY = -22.0         # Testing chamber room gravity
TIME_STEP = 1.0 / 60.0  # 60 FPS physics rate
SUB_STEPS = 2

TOTAL_LEVELS = 20
LEVEL_TIME_LIMIT = 18.0  # 18s time limit per 3D room puzzle
ROOM_LENGTH = 12.0       # Enclosed Testing Chamber Room Length (m)
ROOM_WIDTH_HALF = 3.0    # Full 3D Room Width (Y in [-3.0, +3.0])
TARGET_X_BASE = 11.8     # Exit Door location inside Right Wall

# 3D MOVEMENT SPEED CONSTRAINTS
BOB_MASS = 1.0
RUN_SPEED = 3.5          # Forward X velocity (m/s)
BACK_SPEED = -3.2        # Backward X velocity (m/s) (Fast escape from locked door!)
LATERAL_SPEED = 3.0      # Left/Right Y velocity (m/s)
JUMP_VELOCITY = 6.2      # Upward Z jump impulse (max jump height ~0.88m)
ACCEL_RATE = 0.35        # Ground movement responsiveness

BOB_HALF_EXTENTS = [0.35, 0.3, 0.45]

# ============================================================================
# 2026 NEXT-GEN SCI-FI MATERIAL PALETTE
# ============================================================================
COLORS = {
    # Bob (2026 High-Gloss Cyber-Orange Chassis with Specular Eyes & LED Belt)
    'bob_body': [1.0, 0.48, 0.05, 1.0],       # Cyber-Orange High-Gloss Finish
    'bob_eye_white': [0.98, 0.98, 0.98, 1.0], # Polished White Specular Eyeball
    'bob_pupil': [0.05, 0.05, 0.05, 1.0],     # Deep Obsidian Pupil
    'bob_led_belt': [0.0, 0.90, 1.0, 1.0],    # Neon Cyan LED Energy Belt
    
    # Pressure Plates (Dark Carbon Steel -> Turns Glowing Neon Emerald Green!)
    'plate_grey': [0.22, 0.24, 0.28, 1.0],    # Inactive Dark Carbon-Steel Plate
    'plate_grey_edge': [0.95, 0.65, 0.1, 1.0],# Neon Amber Warning Edge Trim
    'plate_green': [0.10, 1.0, 0.40, 1.0],    # Activated Glowing Neon Emerald Green
    
    # 2026 Room Architecture (Satin Amber Walls & Carbon-Slate Floor)
    'room_wall': [0.85, 0.48, 0.10, 1.0],     # Satin Warm Amber Gold Walls
    'room_corner': [0.18, 0.12, 0.08, 1.0],   # Architectural Dark Bronze Wall Bevel Trim
    'ground': [0.12, 0.14, 0.18, 1.0],        # Carbon-Slate Charcoal Floor
    'platform_body': [0.12, 0.14, 0.18, 1.0], # Carbon-Slate Test Track
    'grid_line': [0.0, 0.85, 1.0, 1.0],       # Ultra-Thin Laser Cyan LED Grid Lines
    
    # 2026 Floating HUD Boards
    'hud_bg': [0.05, 0.08, 0.12, 0.90],       # Obsidian Glass HUD Background
    'hud_frame': [0.0, 0.90, 1.0, 1.0],      # Neon Cyan LED Glass Frame
    
    # Obstacles & Pillars
    'obstacle_body_1': [0.18, 0.20, 0.24, 1.0], # Metallic Dark Titanium Pillar
    'obstacle_body_2': [0.14, 0.16, 0.20, 1.0], # Obsidian Charcoal Pillar
    'obstacle_edge': [0.85, 0.88, 0.92, 1.0],   # Polished Silver Chrome Edge Trim
    
    # Doorway, Barrier Panel & Animated Exit System
    'door_panel': [0.10, 0.12, 0.15, 1.0],    # Heavy Closed Door Barrier Panel
    'door_cutout': [0.03, 0.03, 0.04, 1.0],   # Deep Recessed Dark Doorway Interior
    'door_trim': [0.35, 0.25, 0.15, 1.0],     # Recessed Door Frame Trim
    'door_red_sign': [1.0, 0.10, 0.10, 1.0],  # Neon Red LOCKED Exit Sign
    'door_green_sign': [0.10, 1.0, 0.35, 1.0],# Neon Green UNLOCKED Exit Sign
    'door_cleared': [0.10, 1.0, 0.30, 1.0],   # Deactivated Portal Glow
}

# Default Camera Setup (Exact Perspective Match)
DEFAULT_CAM_YAW = 75.0
DEFAULT_CAM_PITCH = -18.0
DEFAULT_CAM_DIST = 10.5
DEFAULT_CAM_TARGET = [6.0, 0.0, 1.5]

# ============================================================================
# REINFORCEMENT LEARNING HYPERPARAMETERS
# ============================================================================
STATE_DIM = 16       # 16-dimensional 3D sensory observation vector
ACTION_DIM = 5       # 0: -X (Back), 1: +X (Forward), 2: -Y (Left), 3: +Y (Right), 4: +Z (Jump)
GAMMA = 0.98         # Discount factor
LR = 0.0003          # Adam learning rate
BUFFER_SIZE = 100000 # Replay memory capacity
BATCH_SIZE = 64      # Mini-batch size

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 400
TARGET_UPDATE_FREQ = 400

SAVE_DIR = "bob_saves"
