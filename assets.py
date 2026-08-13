"""
Procedural 3D assets generation and room geometry spawning in PyBullet.
Creates walls, entrance/exit doors, obstacles, pressure plates, and Bob's visual components.
"""

import math
import random
import pybullet as p
import config


def create_room_geometry(wall_color, corner_color, floor_color, grid_color):
    """Spawns walls, floor, grid lines, bevels, and perimeter glass guardrails."""
    # Flat ground
    ground_id = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.40])
    p.changeVisualShape(ground_id, -1, rgbaColor=[0.05, 0.06, 0.08, 1.0], specularColor=[0.2, 0.2, 0.2])
    p.changeDynamics(ground_id, -1, lateralFriction=0.1)

    grid_lines = []
    # Ultra-Thin Laser Cyan Tile Grid Lines
    for i in range(0, 13):
        line_id = p.addUserDebugLine([i, -3.0, 0.001], [i, 3.0, 0.001], grid_color, 2.0)
        grid_lines.append(line_id)
    for y in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        line_id = p.addUserDebugLine([0, y, 0.001], [12, y, 0.001], grid_color, 1.8)
        grid_lines.append(line_id)

    # Elevated Chamber Room Floor Track
    platform_length = 6.0
    platform_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[platform_length, 3.0, 0.15]),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[platform_length, 3.0, 0.15],
            rgbaColor=floor_color,
            specularColor=[0.6, 0.6, 0.6]
        ),
        basePosition=[6.0, 0, 0.0]
    )
    p.changeDynamics(platform_id, -1, lateralFriction=0.1)

    guardrail_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[6.0, 0.04, 0.12],
            rgbaColor=[0.0, 0.85, 1.0, 0.50],
            specularColor=[1.0, 1.0, 1.0]
        ),
        basePosition=[6.0, -2.96, 0.27]
    )

    back_wall_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[6.0, 0.1, 2.2]),
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
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[6.0, 0.1, 2.2]),
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

    left_wall_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 3.0, 2.2]),
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

    right_wall_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 3.0, 2.2]),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.1, 3.0, 2.2],
            rgbaColor=wall_color,
            specularColor=[0.5, 0.4, 0.3]
        ),
        basePosition=[12.1, 0, 2.2]
    )

    return ground_id, platform_id, grid_lines


def spawn_bob():
    """Spawns Bob, neon energy belt, and googly eyes in PyBullet."""
    spawn_x = 0.8  # Clear spawn room away from entrance door frame
    spawn_y = 0.0
    
    bob_body_id = p.createMultiBody(
        baseMass=config.BOB_MASS,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=config.BOB_HALF_EXTENTS),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=config.BOB_HALF_EXTENTS,
            rgbaColor=config.COLORS['bob_body'],
            specularColor=[0.9, 0.9, 0.9]
        ),
        basePosition=[spawn_x, spawn_y, 0.6]
    )

    p.changeDynamics(
        bob_body_id, -1,
        mass=config.BOB_MASS,
        lateralFriction=0.1,
        spinningFriction=0.0,
        rollingFriction=0.0,
        linearDamping=0.0,
        angularDamping=1.0,
        restitution=0.0
    )

    # Neon Cyan LED Energy Belt
    belt_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.36, 0.31, 0.04],
        rgbaColor=config.COLORS['bob_led_belt'],
        specularColor=[1.0, 1.0, 1.0]
    )
    belt_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=belt_vis, basePosition=[spawn_x, spawn_y, 0.60])

    # Cute googly eyes
    eye_radius = 0.075
    pupil_radius = 0.038
    eye_ids = []

    eye_left_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=eye_radius,
        rgbaColor=config.COLORS['bob_eye_white'],
        specularColor=[1.0, 1.0, 1.0]
    )
    eye_left_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=eye_left_vis, basePosition=[spawn_x + 0.35, spawn_y - 0.11, 0.72])
    eye_ids.append({'id': eye_left_id, 'offset': [0.35, -0.11, +0.12]})

    eye_right_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=eye_radius,
        rgbaColor=config.COLORS['bob_eye_white'],
        specularColor=[1.0, 1.0, 1.0]
    )
    eye_right_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=eye_right_vis, basePosition=[spawn_x + 0.35, spawn_y + 0.11, 0.72])
    eye_ids.append({'id': eye_right_id, 'offset': [0.35, +0.11, +0.12]})

    pupil_left_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=pupil_radius,
        rgbaColor=config.COLORS['bob_pupil']
    )
    pupil_left_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=pupil_left_vis, basePosition=[spawn_x + 0.41, spawn_y - 0.11, 0.72])
    eye_ids.append({'id': pupil_left_id, 'offset': [0.41, -0.11, +0.12]})

    pupil_right_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=pupil_radius,
        rgbaColor=config.COLORS['bob_pupil']
    )
    pupil_right_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=pupil_right_vis, basePosition=[spawn_x + 0.41, spawn_y + 0.11, 0.72])
    eye_ids.append({'id': pupil_right_id, 'offset': [0.41, +0.11, +0.12]})

    return bob_body_id, belt_id, eye_ids


def create_exit_door(door_x, door_y):
    """Spawns exit door frame, portal target target, and sliding door panel."""
    # Exit Door Trim Frame
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.06, 0.48, 0.73],
            rgbaColor=config.COLORS['door_trim'],
            specularColor=[0.5, 0.4, 0.3]
        ),
        basePosition=[11.99, door_y, 0.70]
    )
    # Door Cutout Glass Panel
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.05, 0.44, 0.69],
            rgbaColor=config.COLORS['door_cutout']
        ),
        basePosition=[11.98, door_y, 0.70]
    )
    # Sliding Physical Barrier Panel
    door_panel_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.06, 0.45, 0.70]),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.06, 0.45, 0.70],
            rgbaColor=config.COLORS['door_panel'],
            specularColor=[0.7, 0.7, 0.7]
        ),
        basePosition=[door_x - 0.05, door_y, 0.80]
    )
    # Exit Goal Portal Target
    target_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.30, 0.50, 0.03]),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.30, 0.50, 0.03],
            rgbaColor=config.COLORS['plate_green'],
            specularColor=[0.8, 1.0, 0.8]
        ),
        basePosition=[door_x + 0.15, door_y, 0.16]
    )

    exit_sign_id = p.addUserDebugText(
        "LOCKED", [11.88, door_y - 0.35, 1.55],
        textColorRGB=config.COLORS['door_red_sign'][:3],
        textSize=1.0, lifeTime=0
    )

    return door_panel_id, target_id, exit_sign_id


def procedural_generate_level(level, target_x, target_y):
    """Spawns strategic layout of pillars and pressure plates depending on level."""
    obstacles = []
    pressure_plates = []

    def add_flat_pressure_plate(x, y, z=0.16, half_extents=None):
        if half_extents is None:
            half_extents = [0.45, 0.55, 0.03]
        
        plate_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents),
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
        pressure_plates.append({
            'id': plate_id,
            'pos': [x, y, z],
            'half_extents': half_extents,
            'activated': False
        })

    def add_obstacle_pillar_with_plate(x, y, half_height):
        # Determine body color based on obstacle index
        color = config.COLORS['obstacle_body_1'] if (len(obstacles) % 2 == 0) else config.COLORS['obstacle_body_2']
        
        obs_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.40, 0.55, half_height]),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.40, 0.55, half_height],
                rgbaColor=color,
                specularColor=[0.5, 0.5, 0.5]
            ),
            basePosition=[x, y, 0.15 + half_height]
        )
        obstacles.append(obs_id)

        # Chrome Trim Edge
        chrome_id = p.createMultiBody(
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
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.38, 0.52, 0.03]),
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

        pressure_plates.append({
            'id': plate_id,
            'pos': [x, y, plate_z],
            'half_extents': [0.38, 0.52, 0.03],
            'activated': False
        })

    # Execute specific room stage curriculum layouts
    if level == 1:
        add_flat_pressure_plate(x=6.0, y=-1.8)
    elif level == 2:
        add_flat_pressure_plate(x=4.0, y=1.8)
        add_flat_pressure_plate(x=8.0, y=-1.8)
    elif level == 3:
        add_obstacle_pillar_with_plate(x=6.0, y=1.5, half_height=0.35)
    elif level == 4:
        add_flat_pressure_plate(x=3.5, y=-1.8)
        add_obstacle_pillar_with_plate(x=7.0, y=1.8, half_height=0.30)
        add_flat_pressure_plate(x=9.5, y=-1.5)
    else:
        # Dynamic Procedural Layout for Advanced Levels (Levels 5-20)
        num_plates = min(2 + (level // 3), 5)
        usable_start = 2.5
        usable_end = target_x - 1.5
        spacing = (usable_end - usable_start) / (num_plates + 1)
        
        y_sides = [-2.0, 2.0, -1.5, 1.5, 0.0]
        for i in range(num_plates):
            px = usable_start + spacing * (i + 1)
            py = y_sides[i % len(y_sides)]
            
            if (i + level) % 2 == 0:
                half_h = max(0.20, min(0.60, 0.20 + level * 0.018))
                add_obstacle_pillar_with_plate(px, py, half_h)
            else:
                add_flat_pressure_plate(px, py, 0.16, [0.45, 0.55, 0.03])

    return obstacles, pressure_plates
