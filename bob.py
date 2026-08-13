"""
Main Launcher for Bob's Experimental Testing Facility.
Launches 2D Desktop GUI Application Interface (`app.py`),
or dedicated PyBullet physics processes (--visual, --headless, --demo).
"""

import sys
import os
import time

def run_session(mode="visual", level=1):
    from utils import TimeManager
    import config
    from agent import DQNAgent
    from environment import BobsWorld3D
    
    render = (mode != "headless")
    time_mgr = TimeManager()
    env = BobsWorld3D(render=render, time_manager=time_mgr)
    agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
    
    print(f"\n==================================================")
    print(f"  BOB'S TESTING FACILITY - Mode: {mode.upper()} | Stage: {level:02d}")
    print(f"==================================================\n")
    
    episodes = 50 if mode == "headless" else 20
    for ep in range(episodes):
        obs, info = env.reset(level=level)
        total_reward = 0
        steps = 0
        
        while True:
            on_ground = env.on_ground
            eval_mode = (mode == "demo")
            action = agent.act(obs, stuck=(env.stuck_counter > 40), on_ground=on_ground, eval_mode=eval_mode)
            next_obs, reward, done, truncated, info = env.step(action)
            
            if mode != "demo":
                agent.memory.push(obs, action, reward, next_obs, done)
                agent.train_step()
                
            obs = next_obs
            total_reward += reward
            steps += 1
            
            if render:
                time.sleep(0.015)
                
            if done:
                if mode != "demo":
                    agent.update_epsilon()
                status_str = "PASSED SUCCESS!" if info.get('success') else f"FAILED ({info.get('failure_reason')})"
                print(f"  [Ep {ep+1}/{episodes}] Stage {level:02d} | {status_str} | Reward: {total_reward:.1f} | Steps: {steps}")
                break
                
    env.close()

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
