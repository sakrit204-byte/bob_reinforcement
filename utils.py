"""
Utility functions and management classes for Bob's World 3D.
Provides TimeManager, WeightManager, and CameraConfigManager for recording and persisting camera parameters.
"""

import time
import os
import json
from datetime import datetime
import torch
import config

class TimeManager:
    """Manages stage timers, remaining time calculations, and best completion records."""
    def __init__(self, level_time_limit=config.LEVEL_TIME_LIMIT):
        self.level_time_limit = level_time_limit
        self.level_best_times = {}
        self.current_level_start_time = 0
        
    def start_level_timer(self):
        self.current_level_start_time = time.time()
    
    def get_remaining_time(self):
        elapsed = time.time() - self.current_level_start_time
        return max(0.0, self.level_time_limit - elapsed)
    
    def get_elapsed_time(self):
        return time.time() - self.current_level_start_time
    
    def is_time_up(self):
        return self.get_remaining_time() <= 0
    
    def update_best_time(self, level, completion_time):
        if level not in self.level_best_times:
            self.level_best_times[level] = completion_time
        else:
            self.level_best_times[level] = min(self.level_best_times[level], completion_time)


class WeightManager:
    """Handles PyTorch model weight saving, loading, metadata tracking, and listing."""
    def __init__(self, save_dir=config.SAVE_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/auto", exist_ok=True)
        os.makedirs(f"{save_dir}/manual", exist_ok=True)
        self.metadata = {}
        self.load_metadata()
    
    def load_metadata(self):
        path = os.path.join(self.save_dir, "metadata.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
    
    def save_metadata(self):
        path = os.path.join(self.save_dir, "metadata.json")
        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def auto_save(self, model, optimizer, episode, level, epsilon, reward):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_path = f"auto/ep{episode}_lv{level}_{ts}.pth"
        full_path = os.path.join(self.save_dir, rel_path)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode,
            'level': level,
            'epsilon': epsilon,
            'reward': reward,
            'timestamp': ts
        }, full_path)
        
        self.metadata[rel_path] = {
            'episode': episode, 'level': level,
            'epsilon': epsilon, 'reward': reward,
            'timestamp': ts, 'type': 'auto'
        }
        self.save_metadata()
        return full_path
    
    def manual_save(self, model, optimizer, episode, level, epsilon, reward, note=""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_path = f"manual/save_ep{episode}_lv{level}_{ts}.pth"
        full_path = os.path.join(self.save_dir, rel_path)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode,
            'level': level,
            'epsilon': epsilon,
            'reward': reward,
            'note': note,
            'timestamp': ts
        }, full_path)
        
        self.metadata[rel_path] = {
            'episode': episode, 'level': level,
            'epsilon': epsilon, 'reward': reward,
            'note': note, 'timestamp': ts, 'type': 'manual'
        }
        self.save_metadata()
        print(f"\n  [Checkpoint Saved]: {rel_path}")
        return full_path
    
    def load_save(self, filename):
        full_path = os.path.join(self.save_dir, filename)
        if os.path.exists(full_path):
            return torch.load(full_path, weights_only=False)
        return None
    
    def get_latest(self):
        if not self.metadata:
            return None
        latest = max(self.metadata.items(), key=lambda x: x[1]['timestamp'])
        return latest[0]
    
    def list_saves(self):
        saves = {'auto': [], 'manual': []}
        for filename, meta in self.metadata.items():
            save_type = meta.get('type', 'auto')
            if save_type not in saves:
                saves[save_type] = []
            saves[save_type].append({'filename': filename, 'metadata': meta})
        for save_type in saves:
            saves[save_type].sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        return saves


class CameraConfigManager:
    """Persists and loads user camera view state indefinitely across app restarts and resets."""
    def __init__(self, save_dir=config.SAVE_DIR):
        self.save_dir = save_dir
        self.config_path = os.path.join(save_dir, "camera_config.json")
        self.log_path = os.path.join(save_dir, "camera_history.log")
        os.makedirs(save_dir, exist_ok=True)
        
    def save_camera(self, yaw, pitch, distance, target, level=None):
        """Saves current camera view parameters to JSON config and records to log history."""
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = {
                "yaw": round(float(yaw), 2),
                "pitch": round(float(pitch), 2),
                "distance": round(float(distance), 2),
                "target": [round(float(t), 2) for t in target],
                "last_updated": ts,
                "level": level
            }
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
                
            # Log recording entry
            log_line = f"[{ts}] Level {level if level else 'General'} | Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Dist: {distance:.2f}m, Target: {target}\n"
            with open(self.log_path, "a") as f:
                f.write(log_line)
        except Exception:
            pass
            
    def load_camera(self):
        """Loads camera parameters from disk defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    return (
                        data.get("yaw", config.DEFAULT_CAM_YAW),
                        data.get("pitch", config.DEFAULT_CAM_PITCH),
                        data.get("distance", config.DEFAULT_CAM_DIST),
                        data.get("target", config.DEFAULT_CAM_TARGET)
                    )
            except Exception:
                pass
        return config.DEFAULT_CAM_YAW, config.DEFAULT_CAM_PITCH, config.DEFAULT_CAM_DIST, config.DEFAULT_CAM_TARGET
