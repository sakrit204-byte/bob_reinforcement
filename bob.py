"""
Main Launcher for Bob's Experimental Testing Facility.
Automatically launches the 2D Desktop GUI Application Interface (`app.py`),
or CLI menu if --cli flag is passed.
"""

import sys
import os

def main():
    if "--cli" in sys.argv:
        from utils import TimeManager, WeightManager
        import config
        from agent import DQNAgent
        from environment import BobsWorld3D
        
        print("Launching CLI Menu...")
        agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
        env = BobsWorld3D(render=True)
        obs, info = env.reset(level=1)
        print("CLI Environment Initialized successfully!")
    else:
        import tkinter as tk
        from app import BobsFacilityApp
        
        root = tk.Tk()
        app = BobsFacilityApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()
