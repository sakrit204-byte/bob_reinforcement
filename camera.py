"""
Camera controls and viewport interaction management for Bob's World 3D.
Implements Blender-style pan/zoom/rotate controls, snap hotkeys, and automated tracking.
"""

import math
import pybullet as p
from utils import CameraConfigManager


class CameraMode:
    BLENDER_CONTROLS = "Blender Controls (MMB Drag = Rotate | Left Click Drag = Pan | Scroll = Zoom)"
    FOLLOW = "Side-Profile Track View"
    OVERVIEW = "Full Room Overview"


class CameraManager:
    """Manages the PyBullet debug visualizer camera controls, configurations, and hotkey listeners."""

    def __init__(self, initial_level=1):
        self.cam_config_mgr = CameraConfigManager()
        self.camera_mode = CameraMode.BLENDER_CONTROLS
        self.current_level = initial_level
        self.cam_x = 0.5
        self.cam_z = 1.0

        # Load saved camera parameters
        yaw, pitch, dist, target = self.cam_config_mgr.load_camera()
        self.user_cam_yaw = yaw
        self.user_cam_pitch = pitch
        self.user_cam_dist = dist
        self.user_cam_target = target

        # Mouse dragging tracking
        self.prev_mouse_x = 0
        self.prev_mouse_y = 0
        self.is_mmb_down = False
        self.is_lmb_down = False

    def reset_camera_view(self, render_mode):
        """Applies defaults or saved configurations on environment reset."""
        if render_mode and self.camera_mode == CameraMode.BLENDER_CONTROLS:
            p.resetDebugVisualizerCamera(
                self.user_cam_dist, self.user_cam_yaw, self.user_cam_pitch, self.user_cam_target
            )

    def toggle_camera_mode(self):
        """Switches between Blender Controls, Track View, and Room Overview."""
        if self.camera_mode == CameraMode.BLENDER_CONTROLS:
            self.camera_mode = CameraMode.FOLLOW
        elif self.camera_mode == CameraMode.FOLLOW:
            self.camera_mode = CameraMode.OVERVIEW
        else:
            self.camera_mode = CameraMode.BLENDER_CONTROLS
        self.update_camera(force=True, bob_pos=None)
        print(f"\n  [Testing Room Camera Toggled]: {self.camera_mode}")

    def handle_keyboard_events(self, keys, bob_pos):
        """Listens for viewpoint snapping and view toggle keys."""
        # 'C' key toggles camera modes
        if (99 in keys and (keys[99] & p.KEY_WAS_TRIGGERED)) or (67 in keys and (keys[67] & p.KEY_WAS_TRIGGERED)):
            self.toggle_camera_mode()
            return True

        # Snap to Right Side View (Numpad 3 or '3')
        elif (51 in keys and (keys[51] & p.KEY_WAS_TRIGGERED)) or (65435 in keys and (keys[65435] & p.KEY_WAS_TRIGGERED)):
            ref_x = bob_pos[0] if bob_pos else 6.0
            p.resetDebugVisualizerCamera(8.0, 90, 0, [ref_x, 0, 1.0])
            self.user_cam_yaw, self.user_cam_pitch = 90.0, 0.0
            self.cam_config_mgr.save_camera(self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=self.current_level)
            print("\n  [Blender View Snapped & Saved]: Right Side View (Numpad 3)")
            return True
            
        # Snap to Front View (Numpad 1 or '1')
        elif (49 in keys and (keys[49] & p.KEY_WAS_TRIGGERED)) or (65436 in keys and (keys[65436] & p.KEY_WAS_TRIGGERED)):
            ref_x = bob_pos[0] if bob_pos else 6.0
            p.resetDebugVisualizerCamera(8.0, 0, 0, [ref_x, 0, 1.0])
            self.user_cam_yaw, self.user_cam_pitch = 0.0, 0.0
            self.cam_config_mgr.save_camera(self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=self.current_level)
            print("\n  [Blender View Snapped & Saved]: Front View (Numpad 1)")
            return True
            
        # Snap to Top View (Numpad 7 or '7')
        elif (55 in keys and (keys[55] & p.KEY_WAS_TRIGGERED)) or (65429 in keys and (keys[65429] & p.KEY_WAS_TRIGGERED)):
            ref_x = bob_pos[0] if bob_pos else 6.0
            p.resetDebugVisualizerCamera(12.0, 0, -89, [ref_x, 0, 1.0])
            self.user_cam_yaw, self.user_cam_pitch = 0.0, -89.0
            self.cam_config_mgr.save_camera(self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=self.current_level)
            print("\n  [Blender View Snapped & Saved]: Top View (Numpad 7)")
            return True

        return False

    def handle_mouse_events(self):
        """Custom Mouse Controls & Auto-Save Camera Configuration."""
        if self.camera_mode != CameraMode.BLENDER_CONTROLS:
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

    def update_camera(self, force=False, bob_pos=None):
        """Updates camera position dynamically in FOLLOW or OVERVIEW modes."""
        if self.camera_mode == CameraMode.BLENDER_CONTROLS and not force:
            return

        if self.camera_mode == CameraMode.FOLLOW and bob_pos is not None:
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

    def record_end_of_level(self):
        """Saves current viewport camera configurations to disk at level end."""
        try:
            cam = p.getDebugVisualizerCamera()
            if cam and len(cam) >= 12:
                self.user_cam_yaw = cam[8]
                self.user_cam_pitch = cam[9]
                self.user_cam_dist = cam[10]
                self.user_cam_target = list(cam[11])
                
            self.cam_config_mgr.save_camera(
                self.user_cam_yaw, self.user_cam_pitch, self.user_cam_dist, self.user_cam_target, level=self.current_level
            )
            print(f"\n  [CAMERA RECORDED AT END OF LEVEL {self.current_level:02d}]: Yaw={self.user_cam_yaw:.1f}°, Pitch={self.user_cam_pitch:.1f}°, Dist={self.user_cam_dist:.1f}m, Target=[{self.user_cam_target[0]:.1f}, {self.user_cam_target[1]:.1f}, {self.user_cam_target[2]:.1f}]")
            print("  [Saved to disk]: bob_saves/camera_config.json & bob_saves/camera_history.log")
        except Exception as e:
            print(f"  [Camera Recording Note]: {e}")
