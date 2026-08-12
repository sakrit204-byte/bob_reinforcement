"""
Main Launcher for Bob's Experimental Testing Facility.
Automatically launches the 2D Desktop GUI Application Interface (`app.py`),
or CLI menu if --cli flag is passed.
"""

import sys
import os

if __name__ == "__main__":
    if "--cli" in sys.argv:
        # Import CLI controller
        from utils import TimeManager, WeightManager
        import config
        from agent import DQNAgent
        from environment import BobsWorld3D
        
        print("Launching CLI Menu...")
        # Run CLI loop
        agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
        env = BobsWorld3D(render=True)
        obs, info = env.reset(level=1)
        print("CLI Environment Initialized successfully!")
    else:
        # Launch Modern 2D Desktop Application GUI
        import tkinter as tk
        from app import BobsFacilityApp
        
        root = tk.Tk()
        app = BobsFacilityApp(root)
        root.mainloop()
