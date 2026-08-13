"""
Main Launcher for Bob's Experimental Testing Facility.
Launches 2D Desktop GUI Application Interface (`app.py`),
or dedicated PyBullet physics processes (--visual, --headless, --demo).
"""

import sys
import os
import time
import pybullet as p

def run_session(mode="visual", level=1):
    from utils import TimeManager
    import config
    from agent import DQNAgent
    from environment import BobsWorld3D
    
    render = (mode != "headless")
    time_mgr = TimeManager()
    env = BobsWorld3D(render=render, time_manager=time_mgr)
    agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
    env.attach_agent(agent)
    
    print(f"\n==================================================")
    print(f"  BOB'S TESTING FACILITY - Mode: {mode.upper()} | Stage: {level:02d}")
    print(f"  (Press 'N' key anytime in 3D viewport to toggle Live Neural Network Visualizer!)")
    print(f"==================================================\n")
    
    ep = 0
    try:
        while True:
            ep += 1
            obs, info = env.reset(level=level)
            total_reward = 0
            steps = 0
            
            while True:
                on_ground = env.on_ground
                eval_mode = (mode == "demo")
                action = agent.act(obs, stuck=(env.stuck_counter > 40), on_ground=on_ground, eval_mode=eval_mode)
                
                try:
                    next_obs, reward, done, truncated, info = env.step(action)
                except Exception as e:
                    # User closed PyBullet GUI window
                    print(f"\n  [Session Ended]: PyBullet window closed by user.")
                    return
                
                if mode != "demo":
                    agent.memory.push(obs, action, reward, next_obs, done)
                    if steps % 4 == 0:
                        agent.replay()
                    
                obs = next_obs
                total_reward += reward
                steps += 1
                
                if done:
                    if mode != "demo":
                        agent.update_epsilon()
                    is_success = info.get('success', False)
                    status_str = "PASSED SUCCESS!" if is_success else f"FAILED ({info.get('failure_reason')})"
                    print(f"  [Episode {ep:03d}] Stage {level:02d} | {status_str} | Reward: {total_reward:.1f} | Steps: {steps} | Epsilon: {agent.epsilon:.3f}")
                    
                    # AUTOMATIC STAGE CURRICULUM PROGRESSION: Advance level on success!
                    if is_success:
                        old_lvl = level
                        level = min(20, level + 1)
                        print(f"\n  🎉 [STAGE CLEARED!]: Bob completed Stage {old_lvl:02d}! Advancing curriculum to Stage {level:02d}!\n")
                        
                    time.sleep(0.5)  # Brief pause between episodes for smooth visual transition
                    break
                    
            if mode == "headless" and ep >= 100:
                print("  [Headless Training Batch Completed (100 Episodes)].")
                break
                
    except KeyboardInterrupt:
        print("\n  [Session Interrupted by User].")
    finally:
        try:
            env.close()
        except:
            pass

def main():
    level = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--level" and i + 1 < len(sys.argv):
            try:
                level = int(sys.argv[i + 1])
            except:
                pass

    if "--visual" in sys.argv or "--cli" in sys.argv:
        run_session(mode="visual", level=level)
    elif "--headless" in sys.argv:
        run_session(mode="headless", level=level)
    elif "--demo" in sys.argv:
        run_session(mode="demo", level=level)
    else:
        import tkinter as tk
        from app import BobsFacilityApp
        
        root = tk.Tk()
        app = BobsFacilityApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()
