"""
Main Entry Point & Interactive Controller for Bob's Experimental Testing Facility.
Launches training, evaluation, curriculum stage testing, and live camera mode visualization.
"""

import os
import sys
import time

import numpy as np
import torch

import config
from utils import TimeManager, WeightManager
from agent import DQNAgent
from environment import BobsWorld3D, CameraMode

# Curriculum Stage Titles
STAGE_TITLES = {
    1: "Basic Locomotion (Walking)",
    2: "Low Step (0.2m Curb)",
    3: "Single Hurdle (0.35m Barrier)",
    4: "Double Hurdles (Rhythm Jumping)",
    5: "High Barrier (0.55m Height)",
    6: "Testing Room Chamber 06",
    7: "Testing Room Chamber 07",
    8: "Testing Room Chamber 08",
    9: "Testing Room Chamber 09",
    10: "Testing Room Chamber 10",
    11: "Advanced Room Chamber 11",
    12: "Advanced Room Chamber 12",
    13: "Advanced Room Chamber 13",
    14: "Advanced Room Chamber 14",
    15: "Advanced Room Chamber 15",
    16: "Master Room Chamber 16",
    17: "Master Room Chamber 17",
    18: "Master Room Chamber 18",
    19: "Master Room Chamber 19",
    20: "Final Mastery Chamber 20",
}

class TrainingManager:
    """Controls the training workflow, GUI visualization, and interactive CLI menu."""
    def __init__(self, render=True):
        self.render = render
        self.time_mgr = TimeManager()
        self.env = BobsWorld3D(render=self.render, time_manager=self.time_mgr)
        self.agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
        self.weight_mgr = WeightManager()
        
        self.current_level = 1
        self.episode = 0
        self.attempts = 0
        self.running = True
        self.level_times = {}
        self.failure_counts = {}
        
    def rebuild_env(self, render):
        """Re-initializes environment with desired rendering mode."""
        if self.render != render:
            self.env.close()
            self.render = render
            self.env = BobsWorld3D(render=self.render, time_manager=self.time_mgr)

    def menu(self):
        """Displays interactive CLI menu."""
        while self.running:
            os.system('cls' if os.name == 'nt' else 'clear')
            stage_name = STAGE_TITLES.get(self.current_level, f"Chamber {self.current_level:02d}")
            print("\n" + "=" * 60)
            print("    BOB'S EXPERIMENTAL TESTING FACILITY - RL LEARNER")
            print("  Story: Bob is trapped in testing rooms learning to exit.")
            print("=" * 60)
            print(f"\n  Current Room  : Stage {self.current_level:02d} - {stage_name}")
            print(f"  Total Episode : {self.episode}")
            print(f"  Epsilon (Expl): {self.agent.epsilon:.3f}")
            print(f"  Replay Buffer : {len(self.agent.memory)} / {config.BUFFER_SIZE}")
            print(f"  Camera Mode   : {self.env.camera_mode}")
            print("\n  1. Start Training Room (GUI Visualized)")
            print("  2. Accelerated Training (Headless Mode)")
            print("  3. Watch Bob Test Run (Demo / Play Mode)")
            print("  4. Load Saved Checkpoint")
            print("  5. Save Current Checkpoint")
            print("  6. View Room Statistics & Best Completion Times")
            print("  7. Toggle Camera Mode (Track View vs Room Overview)")
            print("  8. Reset Model Weights to Random (Fresh Learning from Scratch)")
            print("  9. Exit Testing Facility")
            
            choice = input("\n  Select Option [1-9]: ").strip()
            
            if choice == '1':
                self.rebuild_env(render=True)
                self.train()
            elif choice == '2':
                self.rebuild_env(render=False)
                self.train()
            elif choice == '3':
                self.rebuild_env(render=True)
                self.play_demo()
            elif choice == '4':
                self.load_menu()
            elif choice == '5':
                self.manual_save()
            elif choice == '6':
                self.show_stats()
            elif choice == '7':
                self.env.toggle_camera_mode()
                input("\n  Press Enter to continue...")
            elif choice == '8':
                self.agent.reset_agent_weights()
                self.current_level = 1
                self.episode = 0
                self.attempts = 0
                input("\n  Press Enter to continue...")
            elif choice == '9':
                self.quit()

    def train(self):
        """Main Reinforcement Learning Training Loop."""
        print("\n  [Training Room Started] - Press Ctrl+C to pause and open menu.")
        stage_name = STAGE_TITLES.get(self.current_level, f"Chamber {self.current_level:02d}")
        print(f"  Starting from Stage {self.current_level} ({stage_name}) | Camera: {self.env.camera_mode}\n")
        
        try:
            while self.current_level <= config.TOTAL_LEVELS and self.running:
                state, info = self.env.reset(level=self.current_level)
                total_reward = 0.0
                done = False
                self.attempts += 1
                self.episode += 1
                
                while not done:
                    is_stuck = self.env.stuck_counter > 50
                    on_ground = self.env.on_ground
                    
                    # Action selection under strict movement rules
                    action = self.agent.act(state, stuck=is_stuck, on_ground=on_ground)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    
                    # Optimization step
                    self.agent.replay(config.BATCH_SIZE)
                    
                self.agent.update_epsilon()
                
                if info.get("success"):
                    elapsed = self.time_mgr.get_elapsed_time()
                    self.level_times[self.current_level] = elapsed
                    sname = STAGE_TITLES.get(self.current_level, f"Chamber {self.current_level:02d}")
                    print(f"  [EXIT DOOR REACHED!] Stage {self.current_level:02d} ({sname}) | Time: {elapsed:.1f}s | Reward: {total_reward:+.1f} | Ep: {self.episode}")
                    
                    self.weight_mgr.auto_save(
                        self.agent.policy_net, self.agent.optimizer,
                        self.episode, self.current_level,
                        self.agent.epsilon, total_reward
                    )
                    
                    self.current_level += 1
                    self.attempts = 0
                    
                    if self.current_level > config.TOTAL_LEVELS:
                        print("\n  *** BOB HAS ESCAPED ALL 20 TESTING ROOMS! ***\n")
                        self.weight_mgr.manual_save(
                            self.agent.policy_net, self.agent.optimizer,
                            self.episode, 20, self.agent.epsilon,
                            total_reward, "FINAL - Bob escaped all 20 testing rooms!"
                        )
                        input("  Press Enter to return to menu...")
                        break
                else:
                    reason = info.get('failure_reason', 'unknown')
                    self.failure_counts[reason] = self.failure_counts.get(reason, 0) + 1
                    sname = STAGE_TITLES.get(self.current_level, f"Chamber {self.current_level:02d}")
                    print(f"  [TRY {self.attempts:02d} FAIL: {reason:<7}] Stage {self.current_level:02d} ({sname}) | Ep: {self.episode} | Epsilon (Exploration): {self.agent.epsilon:.3f}")
                        
                if self.episode % 40 == 0:
                    self.weight_mgr.auto_save(
                        self.agent.policy_net, self.agent.optimizer,
                        self.episode, self.current_level,
                        self.agent.epsilon, total_reward
                    )
                    
        except KeyboardInterrupt:
            print("\n  [Paused]")
            self.pause_menu()

    def play_demo(self):
        """Evaluation Mode: Watch Bob execute current test room with greedy policy."""
        sname = STAGE_TITLES.get(self.current_level, f"Chamber {self.current_level:02d}")
        print("\n  [Demo Run Started] - Press Ctrl+C to exit demo.")
        print(f"  Stage {self.current_level:02d}: {sname} | Press 'C' in window to toggle Camera View.\n")
        
        try:
            while True:
                state, info = self.env.reset(level=self.current_level)
                done = False
                total_reward = 0.0
                
                while not done:
                    action = self.agent.act(state, eval_mode=True)
                    state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    time.sleep(0.005)
                    
                if info.get("success"):
                    print(f"  [DEMO EXIT SUCCESS] Stage {self.current_level:02d} cleared! Reward: {total_reward:+.1f}")
                else:
                    print(f"  [DEMO FAIL] Stage {self.current_level:02d} - {info.get('failure_reason')}")
                    
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  [Exited Demo Mode]")

    def pause_menu(self):
        """Pause options when Ctrl+C is pressed."""
        print("\n  1. Save Checkpoint and Continue Training")
        print("  2. Save Checkpoint and Return to Main Menu")
        print("  3. Continue Training")
        choice = input("  Choice [1-3]: ").strip()
        
        if choice == '1':
            self.manual_save()
            self.train()
        elif choice == '2':
            self.manual_save()
        elif choice == '3':
            self.train()

    def load_menu(self):
        """Displays saved checkpoints."""
        saves = self.weight_mgr.list_saves()
        all_saves = saves['manual'] + saves['auto']
        
        if not all_saves:
            print("  No checkpoints found in bob_saves/")
            input("\n  Press Enter...")
            return
            
        print("\n  Available Room Checkpoints:")
        for i, save in enumerate(all_saves[:12], 1):
            m = save['metadata']
            print(f"  {i:02d}. Stage {m['level']}, Ep {m['episode']}, Epsilon: {m.get('epsilon', 0):.3f}, Type: {m.get('type')}")
            
        try:
            choice = int(input("\n  Select Checkpoint # to load [0 to cancel]: "))
            if 0 < choice <= len(all_saves):
                selected = all_saves[choice - 1]
                ckpt = self.weight_mgr.load_save(selected['filename'])
                if ckpt:
                    self.agent.policy_net.load_state_dict(ckpt['model_state_dict'])
                    self.agent.target_net.load_state_dict(ckpt['model_state_dict'])
                    self.agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    self.agent.epsilon = ckpt.get('epsilon', config.EPSILON_END)
                    self.current_level = ckpt.get('level', 1)
                    self.episode = ckpt.get('episode', 0)
                    print(f"  [Successfully Loaded]: Stage {self.current_level}, Episode {self.episode}")
                    input("\n  Press Enter to continue...")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}")
            time.sleep(1.5)

    def manual_save(self):
        """Creates a manual checkpoint."""
        note = input("  Enter optional checkpoint note: ").strip()
        self.weight_mgr.manual_save(
            self.agent.policy_net, self.agent.optimizer,
            self.episode, self.current_level,
            self.agent.epsilon, 0, note
        )
        input("\n  Press Enter...")

    def show_stats(self):
        """Displays room statistics and record times."""
        sname = STAGE_TITLES.get(self.current_level, f"Chamber {self.current_level:02d}")
        print(f"\n  Current Room  : Stage {self.current_level:02d} ({sname})")
        print(f"  Total Episodes: {self.episode}")
        print(f"  Current Epsilon: {self.agent.epsilon:.4f}")
        print(f"  Replay Memory : {len(self.agent.memory)} / {config.BUFFER_SIZE}")
        
        if self.level_times:
            print("\n  Room Best Times:")
            for lvl, t in sorted(self.level_times.items()):
                title = STAGE_TITLES.get(lvl, f"Chamber {lvl:02d}")
                print(f"    Stage {lvl:02d} ({title}): {t:.2f} seconds")
                
        if self.failure_counts:
            print("\n  Failure Summary:")
            for reason, count in sorted(self.failure_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {reason:<10}: {count} occurrences")
                
        input("\n  Press Enter...")

    def quit(self):
        """Quits application cleanly."""
        print("\n  Saving checkpoint before exit...")
        self.weight_mgr.auto_save(
            self.agent.policy_net, self.agent.optimizer,
            self.episode, self.current_level,
            self.agent.epsilon, 0
        )
        self.running = False
        self.env.close()
        print("  Goodbye!")


def main():
    trainer = TrainingManager(render=True)
    print("\n  [Fresh RL Agent Initialized]: Model started with 100% PURE UNTRAINED RANDOM WEIGHTS (Episode 0).")
    print("  Use Menu Option 4 if you ever wish to load a pre-trained checkpoint save.\n")
    time.sleep(0.8)
    trainer.menu()

if __name__ == "__main__":
    main()
