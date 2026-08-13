"""
Enclosed 3D Testing Chamber Room Environment for Bob's World in PyBullet.
Implements Relative Pressure Plate Sensory Guidance, Integrated Digital Scoreboard HUD,
and 2026 Next-Gen Sci-Fi Aesthetics.
"""

import math
import random
from collections import deque
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

import config
from utils import TimeManager, CameraConfigManager

class CameraMode:
    BLENDER_CONTROLS = "Blender Controls (MMB Drag = Rotate | Left Click Drag = Pan | Scroll = Zoom)"
    FOLLOW = "Side-Profile Track View"
    OVERVIEW = "Full Room Overview"

class BobsWorld3D(gym.Env):
    """
    2026 Next-Gen Sci-Fi Testing Chamber Gym Environment.
    
    Cognitive & Sensory Rules:
    1. Bob receives relative direction vectors (dx, dy) to nearest unactivated pressure plate so Bob can learn navigation.
    2. Integrated Digital Scoreboard HUD: Text cleanly mounted ON the suspended obsidian glass board screen.
    3. Elevated Platform with Front Glass Ledge Guardrail preventing viewer fall perception.
    4. 100% Visually Transparent Front Camera Wall with solid physical collision.
    """
    def __init__(self, render=True, time_manager=None):
        super(BobsWorld3D, self).__init__()
        
        self.render_mode = render
        self.time_manager = time_manager or TimeManager()
        self.cam_config_mgr = CameraConfigManager()
        
        self.current_step = 0
        self.current_level = 1
        self.level_completed = False
        self.stuck_counter = 0
        self.last_bob_pos = [0.5, 0.0, 0.6]
        self.on_ground = True
        
        self.target_x = config.TARGET_X_BASE
        self.target_y = 0.0
        
        # Body Tracking
        self.eye_ids = []
        self.belt_id = None
        
        # Pressure Plate & Door State
        self.pressure_plates = []
        self.all_plates_activated = False
        self.door_opening = False
        self.door_open = False
        self.door_z = 0.80
        self.door_panel = None
        self.exit_sign_id = None
        
        # Discovery State
        self.discovered_door = False
        self.door_locked_bumped = False
        self.discovered_plates = set()
        
        # Load Permanent Recorded Camera State from disk
        self.camera_mode = CameraMode.BLENDER_CONTROLS
        self.cam_x = 0.5
        self.cam_z = 1.0
        yaw, pitch, dist, target = self.cam_config_mgr.load_camera()
        self.user_cam_yaw = yaw
        self.user_cam_pitch = pitch
        self.user_cam_dist = dist
        self.user_cam_target = target
        
        self.prev_mouse_x = 0
        self.prev_mouse_y = 0
        self.is_mmb_down = False
        self.is_lmb_down = False
        
        # Action Space: 5 3D Spatial Actions
        self.action_space = spaces.Discrete(config.ACTION_DIM)
        
        # Observation Space (16-dim 3D sensory vector)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.STATE_DIM,), dtype=np.float32
        )
        
        # PyBullet 2026 HD Setup
        if self.render_mode:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            p.resetDebugVisualizerCamera(
                self.user_cam_dist, self.user_cam_yaw, self.user_cam_pitch, self.user_cam_target
            )
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, config.GRAVITY)
        p.setTimeStep(config.TIME_STEP)
        
        # Objects
        self.bob = None
        self.target = None
        self.obstacles = []
        self.passed_obstacles = set()
        self.ground = None
        self.platform = None
        
        # Debug UI
        self.ui_texts = {}
        self.grid_lines = []

    def reset(self, seed=None, level=None):
        super().reset(seed=seed)
        if level is not None:
            self.current_level = min(max(1, level), 20)
            
        self.current_step = 0
        self.level_completed = False
        self.stuck_counter = 0
        self.last_bob_pos = [0.8, 0.0, 0.6]
        self.on_ground = True
        self.passed_obstacles.clear()
        
        # Reset Body, Pressure Plate, Door & Discovery Flags
        self.eye_ids.clear()
        self.belt_id = None
        self.pressure_plates.clear()
        self.all_plates_activated = False
        self.door_opening = False
        self.door_open = False
        self.door_z = 0.80
        self.discovered_door = False
        self.door_locked_bumped = False
        self.discovered_plates.clear()
        
        self.time_manager.start_level_timer()
        p.resetSimulation()
        p.setGravity(0, 0, config.GRAVITY)
        p.setTimeStep(config.TIME_STEP)
        
        self._create_environment()
        target_pos = self._generate_level(self.current_level)
        self._create_bob()
        self._create_exit_door(target_pos)
        self._create_ui()
        
        # Apply recorded camera view defaults
        if self.render_mode and self.camera_mode == CameraMode.BLENDER_CONTROLS:
            p.resetDebugVisualizerCamera(
                self.user_cam_dist, self.user_cam_yaw, self.user_cam_pitch, self.user_cam_target
            )
            
        observation = self._get_observation()
        info = {"level": self.current_level, "time_limit": config.LEVEL_TIME_LIMIT}
        return observation, info

    def _create_environment(self):
        """Creates 2026 Next-Gen Satin Amber Enclosed Room with Laser Cyan Tile Grid Lines & Digital Scoreboard."""
        wall_color = config.COLORS['room_wall']       # Satin Warm Amber Gold [0.85, 0.48, 0.10]
        corner_color = config.COLORS['room_corner']   # Bronze Trim
        floor_color = config.COLORS['ground']         # Carbon-Slate Charcoal [0.12, 0.14, 0.18]
        grid_color = config.COLORS['grid_line']       # Laser Cyan LED [0.0, 0.85, 1.0]
        
        self.ground = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.40])
        p.changeVisualShape(self.ground, -1, rgbaColor=[0.05, 0.06, 0.08, 1.0], specularColor=[0.2, 0.2, 0.2])
        p.changeDynamics(self.ground, -1, lateralFriction=0.1)
        
        self.grid_lines.clear()
        
        # 2026 Ultra-Thin Laser Cyan Tile Grid Lines
        for i in range(0, 13):
            line_id = p.addUserDebugLine(
                [i, -3.0, 0.001], [i, 3.0, 0.001],
                grid_color, 2.0
            )
            self.grid_lines.append(line_id)
        for y in np.linspace(-3.0, 3.0, 13):
            line_id = p.addUserDebugLine(
                [0, y, 0.001], [12, y, 0.001],
                grid_color, 1.8
            )
            self.grid_lines.append(line_id)
            
        # Elevated Chamber Room Floor Track (Elevated 0.15m Depth with Chamfered Edges)
        platform_length = 6.0
        self.platform = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[platform_length, 3.0, 0.15]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[platform_length, 3.0, 0.15],
                rgbaColor=floor_color,
                specularColor=[0.6, 0.6, 0.6]
            ),
            basePosition=[6.0, 0, 0.0]
        )
        p.changeDynamics(self.platform, -1, lateralFriction=0.1)
        
        # ELEVATED PERIMETER GLASS SAFETY GUARDRAIL (y = -2.96, height = 0.18m)
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[6.0, 0.04, 0.12],
                rgbaColor=[0.0, 0.85, 1.0, 0.50],
                specularColor=[1.0, 1.0, 1.0]
            ),
            basePosition=[6.0, -2.96, 0.27]
        )
        
        # Enclosed Back Wall (y = +3.0)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[6.0, 0.1, 2.2]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[6.0, 0.1, 2.2],
                rgbaColor=wall_color,
                specularColor=[0.5, 0.4, 0.3]
            ),
            basePosition=[6.0, 3.0, 2.2]
        )
        
        # Enclosed Front Wall (y = -3.0) - SOLID PHYSICAL COLLISION, 100% VISUALLY TRANSPARENT!
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[6.0, 0.1, 2.2]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[6.0, 0.1, 2.2],
                rgbaColor=[0.85, 0.48, 0.10, 0.0],
                specularColor=[0.0, 0.0, 0.0]
            ),
            basePosition=[6.0, -3.0, 2.2]
        )
        
        # Architectural Dark Bronze Corner Bevels
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[6.0, 0.06, 0.05],
                rgbaColor=corner_color,
                specularColor=[0.6, 0.5, 0.4]
            ),
            basePosition=[6.0, 2.94, 0.05]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[6.0, 0.06, 0.05],
                rgbaColor=corner_color,
                specularColor=[0.6, 0.5, 0.4]
            ),
            basePosition=[6.0, -2.94, 0.05]
        )
        
        # Left Entrance Wall (x = -0.1)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.1, 3.0, 2.2]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.1, 3.0, 2.2],
                rgbaColor=wall_color,
                specularColor=[0.5, 0.4, 0.3]
            ),
            basePosition=[-0.1, 0, 2.2]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.06, 0.48, 0.73],
                rgbaColor=config.COLORS['door_trim'],
                specularColor=[0.5, 0.4, 0.3]
            ),
            basePosition=[0.01, 0, 0.70]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.05, 0.44, 0.69],
                rgbaColor=config.COLORS['door_cutout']
            ),
            basePosition=[0.02, 0, 0.70]
        )
        
        # Right Exit Wall (x = 12.1)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.1, 3.0, 2.2]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.1, 3.0, 2.2],
                rgbaColor=wall_color,
                specularColor=[0.5, 0.4, 0.3]
            ),
            basePosition=[12.1, 0, 2.2]
        )
        
        # 2026 INTEGRATED DIGITAL SCOREBOARD OBSIDIAN GLASS PANEL (Mounted in Front of Back Wall at y = 2.65m)
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[4.8, 0.04, 0.52],
                rgbaColor=config.COLORS['hud_bg'],
                specularColor=[0.9, 0.9, 0.9]
            ),
            basePosition=[6.0, 2.65, 4.25]
        )
        # Neon Cyan LED Glass Board Frame Borders
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[4.82, 0.05, 0.02],
                rgbaColor=config.COLORS['hud_frame'],
                specularColor=[1.0, 1.0, 1.0]
            ),
            basePosition=[6.0, 2.64, 4.77]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[4.82, 0.05, 0.02],
                rgbaColor=config.COLORS['hud_frame'],
                specularColor=[1.0, 1.0, 1.0]
            ),
            basePosition=[6.0, 2.64, 3.73]
        )

    def _create_bob(self):
        """Creates 2026 Cyber-Orange Bob with LED Energy Belt & Specular Googly Eyes."""
        spawn_x = 0.8  # Clear spawn room away from entrance door frame
        spawn_y = 0.0
        self.bob = p.createMultiBody(
            baseMass=config.BOB_MASS,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=config.BOB_HALF_EXTENTS
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=config.BOB_HALF_EXTENTS,
                rgbaColor=config.COLORS['bob_body'],
                specularColor=[0.8, 0.8, 0.8]
            ),
            basePosition=[spawn_x, spawn_y, 0.6]
        )
        
        p.changeDynamics(
            self.bob, -1,
            mass=config.BOB_MASS,
            lateralFriction=0.1,
            spinningFriction=0.01,
            rollingFriction=0.01,
            linearDamping=0.0,
            angularDamping=1.0,
            restitution=0.0
        )
        
        # 2026 NEON CYAN LED ENERGY BELT
        belt_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.36, 0.31, 0.04],
            rgbaColor=config.COLORS['bob_led_belt'],
            specularColor=[1.0, 1.0, 1.0]
        )
        self.belt_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=belt_vis, basePosition=[spawn_x, spawn_y, 0.60])
        
        # CUTE DYNAMIC GOOGLY EYES ON BOB'S FRONT FACE
        eye_radius = 0.075
        pupil_radius = 0.038
        self.eye_ids.clear()
        
        eye_left_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=eye_radius,
            rgbaColor=config.COLORS['bob_eye_white'],
            specularColor=[1.0, 1.0, 1.0]
        )
        eye_left_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=eye_left_vis, basePosition=[spawn_x + 0.35, spawn_y - 0.11, 0.72])
        self.eye_ids.append({'id': eye_left_id, 'offset': [0.35, -0.11, +0.12]})
        
        eye_right_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=eye_radius,
            rgbaColor=config.COLORS['bob_eye_white'],
            specularColor=[1.0, 1.0, 1.0]
        )
        eye_right_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=eye_right_vis, basePosition=[spawn_x + 0.35, spawn_y + 0.11, 0.72])
        self.eye_ids.append({'id': eye_right_id, 'offset': [0.35, +0.11, +0.12]})
        
        pupil_left_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=pupil_radius,
            rgbaColor=config.COLORS['bob_pupil']
        )
        pupil_left_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=pupil_left_vis, basePosition=[spawn_x + 0.41, spawn_y - 0.11, 0.72])
        self.eye_ids.append({'id': pupil_left_id, 'offset': [0.41, -0.11, +0.12]})
        
        pupil_right_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=pupil_radius,
            rgbaColor=config.COLORS['bob_pupil']
        )
        pupil_right_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=pupil_right_vis, basePosition=[spawn_x + 0.41, spawn_y + 0.11, 0.72])
        self.eye_ids.append({'id': pupil_right_id, 'offset': [0.41, +0.11, +0.12]})

    def _sync_bob_eyes(self):
        """Dynamically syncs Bob's googly eyes & LED energy belt to Bob's body on every frame!"""
        if self.bob is None:
            return
        bob_pos, _ = p.getBasePositionAndOrientation(self.bob)
        bx, by, bz = bob_pos[0], bob_pos[1], bob_pos[2]
        
        if self.belt_id is not None:
            p.resetBasePositionAndOrientation(self.belt_id, [bx, by, bz], [0, 0, 0, 1])
            
        for eye_item in self.eye_ids:
            ox, oy, oz = eye_item['offset']
            p.resetBasePositionAndOrientation(
                eye_item['id'],
                [bx + ox, by + oy, bz + oz],
                [0, 0, 0, 1]
            )

    def _create_exit_door(self, position):
        """Creates Heavy Closed Door Barrier Panel & Inner Exit Portal Target at dynamic Y location."""
        door_x = position[0]
        door_y = position[1]
        
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.06, 0.48, 0.73],
                rgbaColor=config.COLORS['door_trim'],
                specularColor=[0.5, 0.4, 0.3]
            ),
            basePosition=[11.99, door_y, 0.70]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.05, 0.44, 0.69],
                rgbaColor=config.COLORS['door_cutout']
            ),
            basePosition=[11.98, door_y, 0.70]
        )
        
        self.door_panel = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.06, 0.45, 0.70]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.06, 0.45, 0.70],
                rgbaColor=config.COLORS['door_panel'],
                specularColor=[0.7, 0.7, 0.7]
            ),
            basePosition=[door_x - 0.05, door_y, 0.80]
        )
        
        self.target = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.30, 0.50, 0.03]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.30, 0.50, 0.03],
                rgbaColor=config.COLORS['plate_green'],
                specularColor=[0.8, 1.0, 0.8]
            ),
            basePosition=[door_x + 0.15, door_y, 0.16]
        )
        
        self.exit_sign_id = p.addUserDebugText(
            "LOCKED", [11.88, door_y - 0.42, 1.65],
            textColorRGB=config.COLORS['door_red_sign'][:3],
            textSize=2.6, lifeTime=0
        )

    def _generate_level(self, level):
        """Generates Strategic, Incremental 3D Room Layouts (Deterministic per Level!)."""
        self.obstacles.clear()
        self.pressure_plates.clear()
        
        rng = random.Random(level * 54321 + 1001)
        
        self.target_x = config.TARGET_X_BASE
        door_y_opts = [0.0, -1.8, 1.8, -2.2, 2.2, 0.0, -1.2, 1.2]
        self.target_y = door_y_opts[(level - 1) % len(door_y_opts)]
        target_pos = [self.target_x, self.target_y, 0.16]
        
        if level == 1:
            # Stage 1: 1 Floor Pressure Plate at (6.0, -1.8) -> Solvable 3D Navigation!
            self._add_pressure_plate(x=6.0, y=-1.8, z=0.16, half_extents=[0.45, 0.55, 0.03])
            return target_pos
            
        elif level == 2:
            self._add_pressure_plate(x=4.0, y=1.8, z=0.16, half_extents=[0.45, 0.55, 0.03])
            self._add_pressure_plate(x=8.0, y=-1.8, z=0.16, half_extents=[0.45, 0.55, 0.03])
            return target_pos
            
        elif level == 3:
            self._add_obstacle_with_plate(x=6.0, y=1.5, half_height=0.35)
            return target_pos
            
        elif level == 4:
            self._add_pressure_plate(x=3.5, y=-1.8, z=0.16, half_extents=[0.45, 0.55, 0.03])
            self._add_obstacle_with_plate(x=7.0, y=1.8, half_height=0.30)
            self._add_pressure_plate(x=9.5, y=-1.5, z=0.16, half_extents=[0.45, 0.55, 0.03])
            return target_pos
            
        else:
            num_plates = min(2 + (level // 3), 5)
            usable_start = 2.5
            usable_end = self.target_x - 1.5
            spacing = (usable_end - usable_start) / (num_plates + 1)
            
            y_sides = [-2.0, 2.0, -1.5, 1.5, 0.0]
            for i in range(num_plates):
                px = usable_start + spacing * (i + 1)
                py = y_sides[i % len(y_sides)]
                
                if (i + level) % 2 == 0:
                    half_h = max(0.20, min(0.60, 0.20 + level * 0.018))
                    self._add_obstacle_with_plate(px, py, half_h)
                else:
                    self._add_pressure_plate(px, py, 0.16, [0.45, 0.55, 0.03])
                    
            return target_pos

    def _add_pressure_plate(self, x, y, z, half_extents):
        """Adds 2026 Dark Carbon Steel Pressure Plate with Neon Amber Warning Edge."""
        plate_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents,
                rgbaColor=config.COLORS['plate_grey'],
                specularColor=[0.6, 0.6, 0.6]
            ),
            basePosition=[x, y, z]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[half_extents[0] + 0.03, half_extents[1] + 0.03, 0.01],
                rgbaColor=config.COLORS['plate_grey_edge'],
                specularColor=[0.9, 0.9, 0.9]
            ),
            basePosition=[x, y, z - 0.02]
        )
        self.pressure_plates.append({
            'id': plate_id,
            'pos': [x, y, z],
            'half_extents': half_extents,
            'activated': False
        })

    def _add_obstacle_with_plate(self, x, y, half_height):
        """Constructs Metallic Dark Titanium 3D Pillar with Chrome Edge Trim & Pressure Plate!"""
        color = config.COLORS['obstacle_body_1'] if (len(self.obstacles) % 2 == 0) else config.COLORS['obstacle_body_2']
        
        obs = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.40, 0.55, half_height]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.40, 0.55, half_height],
                rgbaColor=color,
                specularColor=[0.5, 0.5, 0.5]
            ),
            basePosition=[x, y, 0.15 + half_height]
        )
        self.obstacles.append(obs)
        
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.41, 0.56, 0.02],
                rgbaColor=config.COLORS['obstacle_edge'],
                specularColor=[0.9, 0.9, 0.9]
            ),
            basePosition=[x, y, 0.15 + 2 * half_height]
        )
        
        plate_z = 0.15 + 2 * half_height + 0.03
        plate_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.38, 0.52, 0.03]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.38, 0.52, 0.03],
                rgbaColor=config.COLORS['plate_grey'],
                specularColor=[0.6, 0.6, 0.6]
            ),
            basePosition=[x, y, plate_z]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.40, 0.54, 0.01],
                rgbaColor=config.COLORS['plate_grey_edge'],
                specularColor=[0.9, 0.9, 0.9]
            ),
            basePosition=[x, y, plate_z - 0.02]
        )
        
        self.pressure_plates.append({
            'id': plate_id,
            'pos': [x, y, plate_z],
            'half_extents': [0.38, 0.52, 0.03],
            'activated': False
        })

    def _update_pressure_plates_and_door(self, bob_pos):
        """Checks pressure plate activations & locked door contact."""
        bx, by, bz = bob_pos[0], bob_pos[1], bob_pos[2]
        
        dist_to_door = math.sqrt((bx - self.target_x)**2 + (by - self.target_y)**2)
        if dist_to_door < 2.5:
            self.discovered_door = True
            
        if self.door_panel is not None and not self.door_open and not self.all_plates_activated:
            contacts = p.getContactPoints(bodyA=self.bob, bodyB=self.door_panel)
            if len(contacts) > 0 and not self.door_locked_bumped:
                self.door_locked_bumped = True
                print("\n  🚪 [LOCKED DOOR BUMP DISCOVERY!]: Bob reached exit door, but door is LOCKED! Remaining plates required!")
                
        for i, plate in enumerate(self.pressure_plates):
            px, py, pz = plate['pos']
            dist_to_plate = math.sqrt((bx - px)**2 + (by - py)**2)
            
            if dist_to_plate < 2.2:
                self.discovered_plates.add(i)
                
            if not plate['activated']:
                hx, hy = plate['half_extents'][0], plate['half_extents'][1]
                if abs(bx - px) <= (hx + 0.25) and abs(by - py) <= (hy + 0.25) and (bz >= pz - 0.15):
                    plate['activated'] = True
                    p.changeVisualShape(
                        plate['id'], -1,
                        rgbaColor=config.COLORS['plate_green'],
                        specularColor=[1.0, 1.0, 1.0]
                    )
                    print(f"\n  [PRESSURE PLATE {i+1}/{len(self.pressure_plates)} ACTIVATED!]: Turned GLOWING NEON GREEN!")
                    
        all_green = all(plate['activated'] for plate in self.pressure_plates)
        
        if all_green and not self.all_plates_activated:
            self.all_plates_activated = True
            self.door_opening = True
            print(f"\n  [ALL PRESSURE PLATES GREEN!]: Triggering Exit Door Opening Animation!")
            
            if self.exit_sign_id is not None:
                try:
                    p.removeUserDebugItem(self.exit_sign_id)
                except:
                    pass
            self.exit_sign_id = p.addUserDebugText(
                "OPEN UNLOCKED", [11.88, self.target_y - 0.42, 1.65],
                textColorRGB=config.COLORS['door_green_sign'][:3],
                textSize=2.8, lifeTime=0
            )

        if self.door_opening and not self.door_open:
            self.door_z += 0.08
            
            if self.door_panel is not None:
                p.resetBasePositionAndOrientation(
                    self.door_panel,
                    [self.target_x - 0.05, self.target_y, self.door_z],
                    [0, 0, 0, 1]
                )
                
            if self.door_z >= 2.80:
                self.door_open = True
                self.door_opening = False
                print("  [DOOR FULLY OPENED!]: Bob can now exit through the door!")

    def _get_raycast_sensors(self, bob_pos):
        """Executes 3 fast 3D spatial raycasts."""
        rx, ry, rz = bob_pos[0], bob_pos[1], bob_pos[2]
        
        ray_starts = [
            [rx + 0.35, ry, rz],
            [rx + 0.35, ry - 0.3, rz],
            [rx + 0.35, ry + 0.3, rz],
        ]
        ray_ends = [
            [rx + 4.0, ry, rz],
            [rx + 3.5, ry - 2.5, rz],
            [rx + 3.5, ry + 2.5, rz],
        ]
        
        results = p.rayTestBatch(ray_starts, ray_ends)
        distances = [res[2] for res in results]
        return distances

    def _get_observation(self):
        """
        Constructs 16-dimensional LOCAL 3D SENSORY DISCOVERY observation vector.
        Contains relative direction vectors (dx_plate, dy_plate) to the nearest unactivated pressure plate!
        """
        bob_pos, _ = p.getBasePositionAndOrientation(self.bob)
        bob_vel, _ = p.getBaseVelocity(self.bob)
        self.on_ground = self._check_on_ground()
        time_rem = self.time_manager.get_remaining_time()
        
        ray_distances = self._get_raycast_sensors(bob_pos)
        
        # Calculate relative direction to closest unactivated pressure plate
        dx_plate = 0.0
        dy_plate = 0.0
        unactive_plates = [p_item for p_item in self.pressure_plates if not p_item['activated']]
        
        if unactive_plates:
            # Sort by 3D distance
            unactive_plates.sort(key=lambda p_item: math.sqrt((bob_pos[0] - p_item['pos'][0])**2 + (bob_pos[1] - p_item['pos'][1])**2))
            target_p = unactive_plates[0]['pos']
            dx_plate = (target_p[0] - bob_pos[0]) / 12.0
            dy_plate = (target_p[1] - bob_pos[1]) / 3.0
        else:
            # All plates green -> Target is the Exit Door!
            dx_plate = (self.target_x - bob_pos[0]) / 12.0
            dy_plate = (self.target_y - bob_pos[1]) / 3.0
            
        door_locked_signal = 1.0 if self.door_locked_bumped else (0.5 if self.discovered_door else 0.0)
        discovered_plates_ratio = (len(self.discovered_plates) / float(len(self.pressure_plates))) if self.pressure_plates else 0.0
        active_plates_ratio = (sum(1 for p_item in self.pressure_plates if p_item['activated']) / float(len(self.pressure_plates))) if self.pressure_plates else 0.0
        
        state = np.array([
            bob_pos[0] / 12.0,                  # 1. Bob local X ratio
            bob_pos[1] / 3.0,                   # 2. Bob local Y ratio
            bob_pos[2] / 5.0,                   # 3. Bob local Z height ratio
            bob_vel[0] / 10.0,                  # 4. Velocity X
            bob_vel[1] / 10.0,                  # 5. Velocity Y
            bob_vel[2] / 10.0,                  # 6. Velocity Z
            1.0 if self.on_ground else 0.0,     # 7. Ground flag
            time_rem / config.LEVEL_TIME_LIMIT, # 8. Time limit ratio
            dx_plate,                           # 9. Relative X direction to unactive plate/door
            dy_plate,                           # 10. Relative Y direction to unactive plate/door
            ray_distances[0],                   # 11. Forward Ray
            ray_distances[1],                   # 12. Left Ray
            ray_distances[2],                   # 13. Right Ray
            door_locked_signal,                 # 14. Locked Door Signal
            discovered_plates_ratio,            # 15. Discovered Plates Ratio
            active_plates_ratio                 # 16. Activated Green Plates Ratio
        ], dtype=np.float32)
        
        return state

    def _check_on_ground(self):
        """Checks if Bob is touching ground or platform surface."""
        if self.bob is None:
            return True
        contacts = p.getContactPoints(bodyA=self.bob)
        for c in contacts:
            if c[7][2] > 0.4:
                return True
        return False

    def _check_target_contact(self):
        """Checks if Bob touches the Exit Door target (ONLY AFTER DOOR IS OPEN!)."""
        if self.bob is None or self.target is None or not self.door_open:
            return False
        contacts = p.getContactPoints(bodyA=self.bob, bodyB=self.target)
        return len(contacts) > 0

    def record_end_of_level_camera(self, level):
        """Records active PyBullet camera view parameters to file at the end of the level."""
        if not self.render_mode:
            return
            
        try:
            cam = p.getDebugVisualizerCamera()
            if cam and len(cam) >= 12:
                self.user_cam_yaw = cam[8]
                self.user_cam_pitch = cam[9]
                self.user_cam_dist = cam[10]
                self.user_cam_target = list(cam[11])
                
            self.cam_config_mgr.save_camera(
                self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=level
            )
            print(f"\n  [CAMERA RECORDED AT END OF LEVEL {level:02d}]: Yaw={self.user_cam_yaw:.1f}°, Pitch={self.user_cam_pitch:.1f}°, Dist={self.user_cam_dist:.1f}m, Target=[{self.user_cam_target[0]:.1f}, {self.user_cam_target[1]:.1f}, {self.user_cam_target[2]:.1f}]")
            print("  [Saved to disk]: bob_saves/camera_config.json & bob_saves/camera_history.log")
        except Exception as e:
            print(f"  [Camera Recording Note]: {e}")

    def toggle_camera_mode(self):
        """Switches between Blender Controls, Track View, and Full Room Overview."""
        if self.camera_mode == CameraMode.BLENDER_CONTROLS:
            self.camera_mode = CameraMode.FOLLOW
        elif self.camera_mode == CameraMode.FOLLOW:
            self.camera_mode = CameraMode.OVERVIEW
        else:
            self.camera_mode = CameraMode.BLENDER_CONTROLS
        self._update_camera(force=True)
        print(f"\n  [Testing Room Camera Toggled]: {self.camera_mode}")

    def _handle_keyboard_events(self):
        """Listens for Blender Numpad Viewport Keys & saves camera state!"""
        if not self.render_mode:
            return
        keys = p.getKeyboardEvents()
        
        if (99 in keys and (keys[99] & p.KEY_WAS_TRIGGERED)) or (67 in keys and (keys[67] & p.KEY_WAS_TRIGGERED)):
            self.toggle_camera_mode()
            
        elif (51 in keys and (keys[51] & p.KEY_WAS_TRIGGERED)) or (65435 in keys and (keys[65435] & p.KEY_WAS_TRIGGERED)):
            p.resetDebugVisualizerCamera(8.0, 90, 0, [self.last_bob_pos[0], 0, 1.0])
            self.user_cam_yaw, self.user_cam_pitch = 90.0, 0.0
            self.cam_config_mgr.save_camera(self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=self.current_level)
            print("\n  [Blender View Snapped & Saved]: Right Side View (Numpad 3)")
            
        elif (49 in keys and (keys[49] & p.KEY_WAS_TRIGGERED)) or (65436 in keys and (keys[65436] & p.KEY_WAS_TRIGGERED)):
            p.resetDebugVisualizerCamera(8.0, 0, 0, [self.last_bob_pos[0], 0, 1.0])
            self.user_cam_yaw, self.user_cam_pitch = 0.0, 0.0
            self.cam_config_mgr.save_camera(self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=self.current_level)
            print("\n  [Blender View Snapped & Saved]: Front View (Numpad 1)")
            
        elif (55 in keys and (keys[55] & p.KEY_WAS_TRIGGERED)) or (65429 in keys and (keys[65429] & p.KEY_WAS_TRIGGERED)):
            p.resetDebugVisualizerCamera(12.0, 0, -89, [self.last_bob_pos[0], 0, 1.0])
            self.user_cam_yaw, self.user_cam_pitch = 0.0, -89.0
            self.cam_config_mgr.save_camera(self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=self.current_level)
            print("\n  [Blender View Snapped & Saved]: Top View (Numpad 7)")

    def _handle_mouse_events(self):
        """Custom Mouse Controls & Auto-Save Camera Config."""
        if not self.render_mode or self.camera_mode != CameraMode.BLENDER_CONTROLS:
            return
            
        mouse_events = p.getMouseEvents()
        moved = False
        
        for e in mouse_events:
            event_type = e[0]
            mx, my = e[1], e[2]
            button_idx = e[3]
            button_state = e[4]
            
            if event_type == 2:  # BUTTON EVENT
                if button_idx == 1:   # MMB
                    self.is_mmb_down = (button_state in [1, 2])
                elif button_idx == 0: # LMB
                    self.is_lmb_down = (button_state in [1, 2])
                    
            if event_type in [1, 2]: # MOVE OR DRAG
                dx = mx - self.prev_mouse_x
                dy = my - self.prev_mouse_y
                self.prev_mouse_x = mx
                self.prev_mouse_y = my
                
                if abs(dx) > 80 or abs(dy) > 80:
                    continue
                    
                if self.is_mmb_down and (dx != 0 or dy != 0):
                    self.user_cam_yaw += dx * 0.45
                    self.user_cam_pitch = max(-89.0, min(89.0, self.user_cam_pitch - dy * 0.45))
                    p.resetDebugVisualizerCamera(
                        self.user_cam_dist, self.user_cam_yaw, self.user_cam_pitch, self.user_cam_target
                    )
                    moved = True
                elif self.is_lmb_down and (dx != 0 or dy != 0):
                    rad_yaw = math.radians(self.user_cam_yaw)
                    self.user_cam_target[0] -= (dx * math.cos(rad_yaw) - dy * math.sin(rad_yaw)) * 0.025
                    self.user_cam_target[2] += dy * 0.025
                    p.resetDebugVisualizerCamera(
                        self.user_cam_dist, self.user_cam_yaw, self.user_cam_pitch, self.user_cam_target
                    )
                    moved = True
                    
        if moved:
            self.cam_config_mgr.save_camera(
                self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=self.current_level
            )

    def _update_camera(self, force=False):
        """In BLENDER_CONTROLS mode, NEVER override visualizer camera during step() or reset()!"""
        if not self.render_mode:
            return
            
        if self.camera_mode == CameraMode.BLENDER_CONTROLS and not force:
            return

        if self.camera_mode == CameraMode.FOLLOW and self.bob is not None:
            bob_pos, _ = p.getBasePositionAndOrientation(self.bob)
            target_x = max(1.5, min(10.5, bob_pos[0]))
            target_z = max(0.6, min(3.0, bob_pos[2]))
            
            self.cam_x += (target_x - self.cam_x) * 0.12
            self.cam_z += (target_z - self.cam_z) * 0.12
            
            p.resetDebugVisualizerCamera(
                cameraDistance=7.5,
                cameraYaw=90,
                cameraPitch=0,
                cameraTargetPosition=[self.cam_x, 0.0, self.cam_z]
            )
        elif self.camera_mode == CameraMode.OVERVIEW or force:
            p.resetDebugVisualizerCamera(
                cameraDistance=10.5,
                cameraYaw=75,
                cameraPitch=-18,
                cameraTargetPosition=[6.0, 0.0, 1.5]
            )

    def _create_ui(self):
        """Initializes 2026 Integrated Digital Scoreboard HUD directly ON the Suspended Glass Board Screen!"""
        if not self.render_mode:
            return
            
        for t_id in self.ui_texts.values():
            if t_id is not None:
                try:
                    p.removeUserDebugItem(t_id)
                except:
                    pass
        self.ui_texts.clear()
        
        themes = getattr(config, 'ROOM_THEMES', {})
        theme = themes.get(self.current_level, {'title': f"STAGE {self.current_level:02d}"})
        title_text = theme.get('title', f"STAGE {self.current_level:02d}").upper()
        
        # Mounted 3D Text directly ON the Suspended Glass Digital Scoreboard Screen Surface
        self.ui_texts['header'] = p.addUserDebugText(
            f"[ 🧪 BOB'S TESTING FACILITY - {title_text} ]", [3.0, 2.60, 4.42],
            textColorRGB=[0.0, 0.95, 1.0], textSize=1.75, lifeTime=0
        )
        self.ui_texts['status'] = p.addUserDebugText(
            "TIME: 18.0s  |  PLATES: 0/1 GREEN  |  DOOR: LOCKED", [2.5, 2.60, 4.02],
            textColorRGB=[0.2, 1.0, 0.5], textSize=1.55, lifeTime=0
        )

    def _update_ui(self):
        """Updates Real-Time Digital HUD Status Directly ON the Suspended Glass Board Screen."""
        if not self.render_mode:
            return
            
        self._handle_keyboard_events()
        self._handle_mouse_events()
        self._update_camera()
        
        remaining = self.time_manager.get_remaining_time()
        
        if remaining > 10.0:
            color = [0.2, 1.0, 0.5]
        elif remaining > 5.0:
            color = [1.0, 0.85, 0.1]
        else:
            color = [1.0, 0.2, 0.2]
            
        if self.ui_texts.get('status'):
            try:
                p.removeUserDebugItem(self.ui_texts['status'])
            except:
                pass
                
        active_count = sum(1 for p_item in self.pressure_plates if p_item['activated'])
        total_count = len(self.pressure_plates)
        door_status_str = "DOOR OPEN" if self.door_open else ("OPENING..." if self.door_opening else ("DOOR LOCKED 🔒" if self.door_locked_bumped else "DOOR LOCKED"))
        
        ui_str = f"TIME: {remaining:.1f}s  |  PLATES: {active_count}/{total_count} GREEN  |  {door_status_str}"
        self.ui_texts['status'] = p.addUserDebugText(
            ui_str, [2.5, 2.60, 4.02],
            textColorRGB=color, textSize=1.55, lifeTime=0
        )

    def step(self, action):
        """Executes action across 5 3D Spatial Actions (-X, +X, -Y, +Y, +Z)."""
        self.current_step += 1
        
        if self.time_manager.is_time_up():
            self.record_end_of_level_camera(level=self.current_level)
            obs = self._get_observation()
            return obs, -30.0, True, False, {
                "success": False,
                "failure_reason": "timeout",
                "level": self.current_level
            }
            
        bob_pos, _ = p.getBasePositionAndOrientation(self.bob)
        current_pos = [bob_pos[0], bob_pos[1], bob_pos[2]]
        
        dist_moved = math.sqrt(
            (current_pos[0] - self.last_bob_pos[0])**2 +
            (current_pos[1] - self.last_bob_pos[1])**2
        )
        
        if dist_moved < 0.02:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
            
        self.last_bob_pos = current_pos
        on_ground = self._check_on_ground()
        
        # 5 3D SPATIAL ACTION EXECUTION LOOP (X, Y, Z)
        for _ in range(config.SUB_STEPS):
            vel, _ = p.getBaseVelocity(self.bob)
            curr_vz = vel[2]
            
            new_vx = 0.0
            new_vy = 0.0
            
            if action == 0:    # Backward (-X)
                new_vx = config.BACK_SPEED
            elif action == 1:  # Forward (+X)
                new_vx = config.RUN_SPEED
            elif action == 2:  # Left (-Y)
                new_vy = -config.LATERAL_SPEED
            elif action == 3:  # Right (+Y)
                new_vy = +config.LATERAL_SPEED
            elif action == 4:  # Jump (+Z, Grounded Only!)
                if on_ground:
                    curr_vz = config.JUMP_VELOCITY
                    new_vx = config.RUN_SPEED * 0.7
                    
            new_vz = curr_vz
            if not on_ground and curr_vz < 0.0:
                new_vz += (config.GRAVITY * 0.4) * config.TIME_STEP
                
            p.resetBaseVelocity(self.bob, linearVelocity=[new_vx, new_vy, new_vz])
            
            if self.stuck_counter > 100:
                p.applyExternalForce(
                    self.bob, -1,
                    [random.uniform(-15, 25), random.uniform(-15, 15), random.uniform(10, 25)],
                    [0, 0, 0], p.WORLD_FRAME
                )
                
            p.stepSimulation()
            
            pos, _ = p.getBasePositionAndOrientation(self.bob)
            bounded_y = max(-2.85, min(2.85, pos[1]))
            if pos[1] != bounded_y:
                p.resetBasePositionAndOrientation(self.bob, [pos[0], bounded_y, pos[2]], [0, 0, 0, 1])
                
        # DYNAMICALLY SYNC BOB'S GOOGLY EYES & LED BELT TO BOB'S MOVING CUBE BODY!
        self._sync_bob_eyes()
        
        # UPDATE PRESSURE PLATES & DOOR SLIDING ANIMATION & LOCKED DOOR CONTACT!
        bob_pos, _ = p.getBasePositionAndOrientation(self.bob)
        self._update_pressure_plates_and_door(bob_pos)
        
        if self.render_mode:
            self._update_ui()
            
        obs = self._get_observation()
        bob_vel, _ = p.getBaseVelocity(self.bob)
        
        reward = dist_moved * 15.0 + max(0.0, bob_vel[0]) * 0.10
        
        # Reward bonus for each activated pressure plate
        for i, plate in enumerate(self.pressure_plates):
            if plate['activated'] and i not in self.passed_obstacles:
                reward += 35.0
                self.passed_obstacles.add(i)
                
        # Small penalty if stuck against locked exit door panel to encourage turning around
        if self.door_locked_bumped and not self.all_plates_activated:
            reward -= 0.50
            
        if self.stuck_counter > 40:
            reward -= 0.25
            
        terminated = False
        target_hit = self._check_target_contact()
        
        if target_hit:
            elapsed = self.time_manager.get_elapsed_time()
            time_bonus = max(0.0, (config.LEVEL_TIME_LIMIT - elapsed) * 6.0)
            reward += 250.0 + time_bonus
            terminated = True
            self.level_completed = True
            self.time_manager.update_best_time(self.current_level, elapsed)
            self.record_end_of_level_camera(level=self.current_level)
            
        elif bob_pos[2] < -0.8:
            reward -= 30.0
            terminated = True
            self.record_end_of_level_camera(level=self.current_level)
            
        elif self.stuck_counter > 280:
            reward -= 20.0
            terminated = True
            self.record_end_of_level_camera(level=self.current_level)
            
        info = {
            "success": self.level_completed,
            "level": self.current_level,
            "time_remaining": self.time_manager.get_remaining_time(),
            "on_ground": on_ground,
            "failure_reason": (
                "stuck" if self.stuck_counter > 280 else
                "fall" if bob_pos[2] < -0.8 else
                "timeout" if self.time_manager.is_time_up() else None
            )
        }
        
        return obs, reward, terminated, False, info

    def close(self):
        p.disconnect()
