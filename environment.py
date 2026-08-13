"""
Enclosed 3D Testing Chamber Room Environment for Bob's World in PyBullet.
Implements gymnasium.Env interface with modular UI, Camera, and Spawning subsystems.
"""

import math
import random
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

import config
from utils import TimeManager
import assets
from camera import CameraManager, CameraMode
from ui import DigitalBoardScreen


class BobsWorld3D(gym.Env):
    """
    Modular Gym Env wrapping PyBullet physics engine for Bob's World puzzle stage clearing.
    Delegates room asset creation, camera views, and blackboard text rendering to sub-modules.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=True, time_manager=None):
        super().__init__()
        self.render_mode = render
        self.time_manager = time_manager if time_manager else TimeManager()
        self.camera_manager = CameraManager(initial_level=1)
        self.agent = None

        # Grid lines tracking
        self.grid_lines = []

        # Permanent Assets
        self.ground = None
        self.platform = None
        self.bob = None
        self.belt_id = None
        self.eye_ids = []
        self.door_panel = None
        self.target = None
        self.exit_sign_id = None

        # Interactive obstacles and plates
        self.obstacles = []
        self.pressure_plates = []

        # HUD UI Generator
        self.board_screen_gen = None
        self.board_body_id = None
        self.show_nn_visualizer = False
        self._nn_hud = None

        # Level layout constraints
        self.current_level = 1
        self.target_x = config.TARGET_X_BASE
        self.target_y = 0.0

        # Action and Observation Spaces
        self.action_space = spaces.Discrete(config.ACTION_DIM)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.STATE_DIM,), dtype=np.float32
        )

        # PyBullet Setup
        if self.render_mode:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            self.camera_manager.reset_camera_view(render_mode=True)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, config.GRAVITY)
        p.setTimeStep(config.TIME_STEP)

        # Initial Build
        self.reset_environment_state()

    def reset_environment_state(self):
        """Cleans up previous scene and spawns base elements."""
        p.resetSimulation()
        # resetSimulation wipes ALL state — must re-apply search path, visualizer, physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.render_mode:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.setGravity(0, 0, config.GRAVITY)
        p.setTimeStep(config.TIME_STEP)

        # Clear active debug lines
        self.grid_lines.clear()
        self.eye_ids.clear()
        self.obstacles.clear()
        self.pressure_plates.clear()

        # Re-build room walls, tracks, entrance geometries
        wall_color = config.COLORS['room_wall']
        corner_color = config.COLORS['room_corner']
        floor_color = config.COLORS['ground']
        grid_color = config.COLORS['grid_line']

        self.ground, self.platform, self.grid_lines = assets.create_room_geometry(
            wall_color, corner_color, floor_color, grid_color
        )

    def attach_agent(self, agent):
        """Links agent to environment for Live Neural Network HUD updates."""
        self.agent = agent

    def reset(self, level=1, seed=None, options=None):
        super().reset(seed=seed)
        self.current_level = level
        self.camera_manager.current_level = level

        # Telemetry tracker variables
        self.stuck_counter = 0
        self.last_bob_pos = [0.8, 0.0, 0.6]
        self.level_completed = False
        self.passed_obstacles = set()
        self.on_ground = True
        self.prev_target_dist = None  # Reset so first step doesn't get stale progress reward

        self.discovered_door = False
        self.door_locked_bumped = False
        self.discovered_plates = set()

        # Reset Door State
        self.door_open = False
        self.door_opening = False
        self.all_plates_activated = False
        self.door_z = 0.80

        # Recreate static room assets & spawn targets
        self.reset_environment_state()

        # Determine Exit Door position
        door_y_opts = [0.0, -1.8, 1.8, -2.2, 2.2, 0.0, -1.2, 1.2]
        self.target_y = door_y_opts[(self.current_level - 1) % len(door_y_opts)]

        # Spawn Door Panel and Target Goal
        self.door_panel, self.target, self.exit_sign_id = assets.create_exit_door(
            self.target_x, self.target_y
        )

        # Spawn obstacles and pressure plates based on stage rules
        self.obstacles, self.pressure_plates = assets.procedural_generate_level(
            self.current_level, self.target_x, self.target_y
        )

        # Spawn Bob cube body and sensors
        self.bob, self.belt_id, self.eye_ids = assets.spawn_bob()

        # Synchronize visual attachments
        self._sync_bob_eyes()

        # Setup Visual UI Screen Blackboard
        self._create_ui()

        # Start the level timer AFTER all construction is done (so building time doesn't eat into Bob's 18s)
        self.time_manager.start_level_timer()

        # Camera positioning
        self.camera_manager.reset_camera_view(render_mode=self.render_mode)
        if self.camera_manager.camera_mode != CameraMode.BLENDER_CONTROLS:
            self._update_camera(force=True)

        observation = self._get_observation()
        info = {"level": self.current_level, "time_limit": config.LEVEL_TIME_LIMIT}
        return observation, info

    def step(self, action):
        if self.bob is None:
            raise RuntimeError("Environment reset() must be called before step().")

        # Telemetry updates
        current_pos, _ = p.getBasePositionAndOrientation(self.bob)
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

        # Listen for keyboard/mouse viewport changes
        if self.render_mode:
            self._handle_keyboard_events()
            self.camera_manager.handle_mouse_events()

        # 5 3D SPATIAL ACTION EXECUTION LOOP (X, Y, Z)
        for _ in range(config.SUB_STEPS):
            vel, _ = p.getBaseVelocity(self.bob)
            curr_vx, curr_vy, curr_vz = vel[0], vel[1], vel[2]
            
            target_vx = 0.0
            target_vy = 0.0
            target_vz = curr_vz
            
            if action == 0:    # Backward (-X)
                target_vx = config.BACK_SPEED
            elif action == 1:  # Forward (+X)
                target_vx = config.RUN_SPEED
            elif action == 2:  # Left (-Y)
                target_vy = -config.LATERAL_SPEED
            elif action == 3:  # Right (+Y)
                target_vy = +config.LATERAL_SPEED
            elif action == 4:  # Jump (+Z, Grounded Only!)
                if on_ground:
                    target_vz = config.JUMP_VELOCITY
                    target_vx = config.RUN_SPEED * 0.6
                    
            if not on_ground and target_vz < 0.0:
                target_vz += (config.GRAVITY * 0.4) * config.TIME_STEP
                
            # Smoothly transition velocities using ACCEL_RATE (no instant snaps)
            accel = config.ACCEL_RATE
            smooth_vx = curr_vx + (target_vx - curr_vx) * accel
            smooth_vy = curr_vy + (target_vy - curr_vy) * accel
            
            # Reset velocity with smooth horizontal components and vertical impulse
            p.resetBaseVelocity(self.bob, linearVelocity=[smooth_vx, smooth_vy, target_vz])
            
            # Completely reset position bounds and lock orientation to [0, 0, 0, 1] on every sub-step!
            pos, _ = p.getBasePositionAndOrientation(self.bob)
            bounded_y = max(-2.85, min(2.85, pos[1]))
            p.resetBasePositionAndOrientation(self.bob, [pos[0], bounded_y, pos[2]], [0, 0, 0, 1])

            if self.stuck_counter > 100:
                p.applyExternalForce(
                    self.bob, -1,
                    [random.uniform(-15, 25), random.uniform(-15, 15), random.uniform(10, 25)],
                    [0, 0, 0], p.WORLD_FRAME
                )
            p.stepSimulation()

        # Update attachments
        self._sync_bob_eyes()
        self.camera_manager.update_camera(bob_pos=current_pos)

        # Update plate activation states and door animation
        bob_pos, _ = p.getBasePositionAndOrientation(self.bob)
        self._update_pressure_plates_and_door(bob_pos)

        if self.render_mode:
            self._update_ui()

        obs = self._get_observation()

        # Calculate reward
        unactive_plates = [p_item for p_item in self.pressure_plates if not p_item['activated']]
        if unactive_plates:
            unactive_plates.sort(key=lambda p_item: math.sqrt((bob_pos[0] - p_item['pos'][0])**2 + (bob_pos[1] - p_item['pos'][1])**2))
            target_p = unactive_plates[0]['pos']
            curr_target_dist = math.sqrt((bob_pos[0] - target_p[0])**2 + (bob_pos[1] - target_p[1])**2)
        else:
            curr_target_dist = math.sqrt((bob_pos[0] - self.target_x)**2 + (bob_pos[1] - self.target_y)**2)
            
        prev = self.prev_target_dist if self.prev_target_dist is not None else curr_target_dist
        dist_progress = prev - curr_target_dist
        self.prev_target_dist = curr_target_dist
        
        # Smooth Positive Attraction Reward + Symmetric Progress (prevents back-and-forth oscillation exploitation)
        attraction_reward = max(0.0, (1.0 - (curr_target_dist / 12.0))) * 1.5
        reward = attraction_reward + (dist_progress * 30.0)
        
        # Reward mega-bonus for each activated pressure plate
        for i, plate in enumerate(self.pressure_plates):
            if plate['activated'] and i not in self.passed_obstacles:
                reward += 100.0
                self.passed_obstacles.add(i)
                
        # Jump penalty
        if action == 4:
            reward -= 1.0
                
        # HEAVY NEURAL REJECTION: Reaching/bumping locked door in the CURRENT step
        is_touching_locked_door = False
        if self.door_panel is not None and not self.door_open and not self.all_plates_activated:
            contacts = p.getContactPoints(bodyA=self.bob, bodyB=self.door_panel)
            if len(contacts) > 0 or bob_pos[0] >= 11.80:
                is_touching_locked_door = True
                if not self.door_locked_bumped:
                    self.door_locked_bumped = True
                    print("\n  🚪 [LOCKED DOOR BUMP DISCOVERY!]: Bob reached exit door, but door is LOCKED! Repelling Bob back to find plates!")
                    
        if is_touching_locked_door:
            reward -= 50.0
            if action == 0:  # Action 0 = Moving Backward (-X) away from door
                reward += 30.0
            elif action == 1: # Action 1 = Moving Forward (+X) into locked door
                reward -= 100.0
            
        if not self.all_plates_activated and bob_pos[0] > 10.0:
            reward -= 2.0  # Continuous penalty for hanging near the exit door without activating plates
            
        if self.stuck_counter > 40:
            reward -= 0.35
            
        # Terminal checks
        terminated = False
        target_hit = self._check_target_contact()
        
        if target_hit:
            elapsed = self.time_manager.get_elapsed_time()
            time_bonus = max(0.0, (config.LEVEL_TIME_LIMIT - elapsed) * 6.0)
            reward += 250.0 + time_bonus
            terminated = True
            self.level_completed = True
            self.time_manager.update_best_time(self.current_level, elapsed)
            self.camera_manager.record_end_of_level()
            
        elif bob_pos[2] < -0.8:
            reward -= 30.0
            terminated = True
            self.camera_manager.record_end_of_level()
            
        elif self.stuck_counter > 280:
            reward -= 20.0
            terminated = True
            self.camera_manager.record_end_of_level()
            
        elif self.time_manager.is_time_up():
            terminated = True
            self.camera_manager.record_end_of_level()

        active_count = sum(1 for p_item in self.pressure_plates if p_item['activated'])
        total_count = len(self.pressure_plates)
        
        failure_detail = None
        if not self.level_completed:
            if bob_pos[2] < -0.8:
                failure_detail = f"Fell Off Ledge (Plates: {active_count}/{total_count})"
            elif self.stuck_counter > 280:
                failure_detail = f"Stuck in Corner/Wall (Plates: {active_count}/{total_count})"
            elif self.time_manager.is_time_up():
                if active_count == 0:
                    failure_detail = f"Timeout: Found 0/{total_count} plates"
                elif active_count < total_count:
                    failure_detail = f"Timeout: Got {active_count}/{total_count} plates (missing {total_count - active_count})"
                else:
                    failure_detail = f"Timeout: All {total_count}/{total_count} plates green, but couldn't reach door in time"
                    
        info = {
            "success": self.level_completed,
            "failure_reason": failure_detail,
            "level": self.current_level,
            "remaining_time": self.time_manager.get_remaining_time(),
            "active_plates": active_count,
            "total_plates": total_count
        }

        # Truncated gymnasium status
        truncated = self.time_manager.is_time_up()

        return obs, reward, terminated, truncated, info

    def _sync_bob_eyes(self):
        """Syncs the visual components (belt, eyes) to Bob's physical position."""
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

    def _get_raycast_sensors(self, bob_pos):
        """Executes 3 spatial raycasts in front of Bob."""
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
        """Constructs 16-dimensional LOCAL observation vector."""
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
            unactive_plates.sort(key=lambda p_item: math.sqrt((bob_pos[0] - p_item['pos'][0])**2 + (bob_pos[1] - p_item['pos'][1])**2))
            target_p = unactive_plates[0]['pos']
            dx_plate = (target_p[0] - bob_pos[0]) / 12.0
            dy_plate = (target_p[1] - bob_pos[1]) / 3.0
        else:
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
        """Checks if Bob touches the exit portal target."""
        if self.bob is None or self.target is None or not self.door_open:
            return False
        contacts = p.getContactPoints(bodyA=self.bob, bodyB=self.target)
        return len(contacts) > 0

    def _update_pressure_plates_and_door(self, bob_pos):
        bx, by, bz = bob_pos[0], bob_pos[1], bob_pos[2]
        
        dist_to_door = math.sqrt((bx - self.target_x)**2 + (by - self.target_y)**2)
        if dist_to_door < 2.5:
            self.discovered_door = True
            
        if self.door_panel is not None and not self.door_open and not self.all_plates_activated:
            contacts = p.getContactPoints(bodyA=self.bob, bodyB=self.door_panel)
            if len(contacts) > 0 or bx >= 11.85:
                if not self.door_locked_bumped:
                    self.door_locked_bumped = True
                    print("\n  🚪 [LOCKED DOOR BUMP DISCOVERY!]: Bob reached exit door, but door is LOCKED! Repelling Bob back to find plates!")
                # Propel Bob backwards away from door
                p.resetBaseVelocity(self.bob, linearVelocity=[-3.0, random.uniform(-1.0, 1.0), 0.5])
                p.applyExternalForce(self.bob, -1, [-30.0, 0.0, 5.0], [0, 0, 0], p.WORLD_FRAME)
                
        for i, plate in enumerate(self.pressure_plates):
            px, py, pz = plate['pos']
            dist_to_plate = math.sqrt((bx - px)**2 + (by - py)**2)
            
            if dist_to_plate < 2.2:
                self.discovered_plates.add(i)
                
            if not plate['activated']:
                dist_to_plate_2d = math.sqrt((bx - px)**2 + (by - py)**2)
                if dist_to_plate_2d <= 0.85 and (bz >= pz - 0.35):
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
            # Update sign display
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

    def _handle_keyboard_events(self):
        """Processes keystroke inputs in PyBullet viewport."""
        keys = p.getKeyboardEvents()
        
        # 'N' toggles live neural HUD
        if (110 in keys and (keys[110] & p.KEY_WAS_TRIGGERED)) or (78 in keys and (keys[78] & p.KEY_WAS_TRIGGERED)):
            self.show_nn_visualizer = not getattr(self, 'show_nn_visualizer', False)
            status_str = "ENABLED (Press 'N' to Hide)" if self.show_nn_visualizer else "DISABLED"
            print(f"\n  [ 🧠 LIVE NEURAL NETWORK HUD ]: Visualizer Window {status_str}")
            
        else:
            bob_pos, _ = p.getBasePositionAndOrientation(self.bob) if self.bob else (None, None)
            self.camera_manager.handle_keyboard_events(keys, bob_pos)

    def _update_camera(self, force=False):
        bob_pos, _ = p.getBasePositionAndOrientation(self.bob) if self.bob else (None, None)
        self.camera_manager.update_camera(force=force, bob_pos=bob_pos)

    def _create_ui(self):
        """Creates the 3D scoreboard blackboard object in room."""
        if not self.render_mode:
            return
            
        if self.board_screen_gen is None:
            self.board_screen_gen = DigitalBoardScreen()
            
        tex_path, _ = self.board_screen_gen.generate(
            stage_num=self.current_level,
            remaining_time=self.time_manager.get_remaining_time(),
            active_plates=0,
            total_plates=len(self.pressure_plates) if hasattr(self, 'pressure_plates') else 1,
            door_status="DOOR LOCKED"
        )
        tex_id = p.loadTexture(tex_path)
        self.board_screen_gen.cached_tex_id = tex_id
        
        self.board_body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[3.2, 0.02, 0.35],
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                specularColor=[0.9, 0.9, 0.9]
            ),
            basePosition=[6.0, 2.85, 4.25]
        )
        p.changeVisualShape(self.board_body_id, -1, textureUniqueId=tex_id)

    def _update_ui(self):
        """Redraws scoreboard UI screen if data has changed."""
        if not self.render_mode or self.board_screen_gen is None:
            return
            
        time_rem = self.time_manager.get_remaining_time()
        active_plates = sum(1 for p_item in self.pressure_plates if p_item['activated'])
        total_plates = len(self.pressure_plates)
        door_status = "DOOR OPEN" if self.door_open else ("OPENING..." if self.door_opening else "DOOR LOCKED")

        tex_path, changed = self.board_screen_gen.generate(
            stage_num=self.current_level,
            remaining_time=time_rem,
            active_plates=active_plates,
            total_plates=total_plates,
            door_status=door_status
        )

        if changed and self.board_body_id is not None:
            try:
                tex_id = p.loadTexture(tex_path)
                self.board_screen_gen.cached_tex_id = tex_id
                p.changeVisualShape(self.board_body_id, -1, textureUniqueId=tex_id)
            except Exception as e:
                print(f"UI Update Warning: {e}")

        # Render 2D Tkinter HUD neural network window if show is toggled
        self._render_nn_visualizer()

    def _render_nn_visualizer(self):
        """Spawns or closes the 2D Neural Network HUD window when 'N' is toggled."""
        if not self.render_mode:
            return

        show = getattr(self, 'show_nn_visualizer', False)

        if show:
            hud = getattr(self, '_nn_hud', None)
            if hud is None or not hud.alive:
                from visualizer import NeuralNetworkHUD
                self._nn_hud = NeuralNetworkHUD(agent_ref=self.agent)
            
            if self._nn_hud.alive:
                self._nn_hud.update_hud()
            else:
                self.show_nn_visualizer = False
                self._nn_hud = None
        else:
            hud = getattr(self, '_nn_hud', None)
            if hud is not None:
                hud.close()
                self._nn_hud = None

    def close(self):
        """Cleans up visual windows and exits client connections."""
        hud = getattr(self, '_nn_hud', None)
        if hud is not None:
            hud.close()
        try:
            p.disconnect(self.client)
        except:
            pass
