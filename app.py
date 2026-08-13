"""
Modern 2D Desktop GUI Application Interface for Bob's World 3D RL Testing Facility.
Features Dark Obsidian Cyber Theme, Facility Card Selector, Live Telemetry Dashboard,
Stage Controls, and Integration with PyBullet 3D RL Physics Engine.
"""

import sys
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# Insert project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from agent import DQNAgent
from environment import BobsWorld3D, CameraMode

class BobsFacilityApp:
    """
    Ultra-Modern 2D Desktop GUI Application Interface.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Bob's Experimental Testing Facility - RL Control Suite v2.0")
        self.root.geometry("980x680")
        self.root.resizable(True, True)
        self.root.configure(bg="#0c1017")
        
        # RL Agent & Env Reference
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM
        self.agent = DQNAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        
        self.selected_facility = "facility_1"
        self.selected_level = 1
        self.is_training = False
        
        self._setup_styles()
        self._build_header()
        self._build_facility_selector()
        self._build_dashboard_controls()
        self._build_telemetry_panel()
        self._build_footer()
        
    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Dark Cyber Theme Palette
        self.style.configure('.', background='#0c1017', foreground='#e0e6ed', font=('Segoe UI', 10))
        self.style.configure('TFrame', background='#0c1017')
        self.style.configure('Header.TLabel', background='#0c1017', foreground='#00f0ff', font=('Segoe UI', 18, 'bold'))
        self.style.configure('SubHeader.TLabel', background='#0c1017', foreground='#8b9bb4', font=('Segoe UI', 10, 'italic'))
        self.style.configure('Card.TFrame', background='#161d2a', relief='flat')
        self.style.configure('CardActive.TFrame', background='#1e293b', relief='solid', borderwidth=2)
        
        self.style.configure('Primary.TButton', background='#00a8ff', foreground='#ffffff', font=('Segoe UI', 11, 'bold'), padding=8)
        self.style.map('Primary.TButton', background=[('active', '#0088cc'), ('disabled', '#334155')])
        
        self.style.configure('Success.TButton', background='#10b981', foreground='#ffffff', font=('Segoe UI', 11, 'bold'), padding=8)
        self.style.map('Success.TButton', background=[('active', '#059669')])
        
        self.style.configure('Warning.TButton', background='#f59e0b', foreground='#ffffff', font=('Segoe UI', 10, 'bold'), padding=6)
        self.style.configure('Danger.TButton', background='#ef4444', foreground='#ffffff', font=('Segoe UI', 10, 'bold'), padding=6)
        
    def _build_header(self):
        header_frame = ttk.Frame(self.root, padding="15 15 15 10")
        header_frame.pack(fill='x')
        
        title_lbl = ttk.Label(header_frame, text="🧪 BOB'S EXPERIMENTAL TESTING FACILITY", style='Header.TLabel')
        title_lbl.pack(anchor='w')
        
        subtitle_lbl = ttk.Label(
            header_frame,
            text="Autonomous Deep Reinforcement Learning Suite  |  2026 Next-Gen 3D Spatial Physics Environment",
            style='SubHeader.TLabel'
        )
        subtitle_lbl.pack(anchor='w', pady=(2, 0))
        
        sep = ttk.Separator(self.root, orient='horizontal')
        sep.pack(fill='x', padx=15, pady=5)

    def _build_facility_selector(self):
        sec_lbl = tk.Label(self.root, text="SELECT TESTING FACILITY & ENVIRONMENT", bg="#0c1017", fg="#00f0ff", font=('Segoe UI', 11, 'bold'))
        sec_lbl.pack(anchor='w', padx=18, pady=(8, 4))
        
        card_container = tk.Frame(self.root, bg="#0c1017")
        card_container.pack(fill='x', padx=15, pady=5)
        
        # Facility Card 1 (ACTIVE)
        card1 = tk.Frame(card_container, bg="#1a2332", highlightbackground="#00f0ff", highlightthickness=2, padx=12, pady=10)
        card1.pack(side='left', expand=True, fill='both', padx=5)
        
        f1_title = tk.Label(card1, text="🧪 Facility 1: Bob's 3D Puzzle Chamber", bg="#1a2332", fg="#ffffff", font=('Segoe UI', 11, 'bold'))
        f1_title.pack(anchor='w')
        f1_desc = tk.Label(
            card1,
            text="• 3D Spatial Navigation (X, Y, Z)\n• Pressure Plate Discovery Puzzle\n• Sliding Barrier Door & Ledge Edge\n• 20 Incremental Stages",
            bg="#1a2332", fg="#94a3b8", font=('Segoe UI', 9), justify='left'
        )
        f1_desc.pack(anchor='w', pady=4)
        f1_badge = tk.Label(card1, text="ACTIVE FACILITY", bg="#059669", fg="#ffffff", font=('Segoe UI', 8, 'bold'), padx=6, pady=2)
        f1_badge.pack(anchor='e', pady=(4, 0))
        
        # Facility Card 2 (LOCKED)
        card2 = tk.Frame(card_container, bg="#111827", highlightbackground="#374151", highlightthickness=1, padx=12, pady=10)
        card2.pack(side='left', expand=True, fill='both', padx=5)
        
        f2_title = tk.Label(card2, text="🌀 Facility 2: Quantum Labyrinth", bg="#111827", fg="#6b7280", font=('Segoe UI', 11, 'bold'))
        f2_title.pack(anchor='w')
        f2_desc = tk.Label(
            card2,
            text="• Multi-Chamber Labyrinth Paths\n• Dynamic Moving Laser Barriers\n• Teleportation Gateways\n• Multi-Agent Collaboration",
            bg="#111827", fg="#4b5563", font=('Segoe UI', 9), justify='left'
        )
        f2_desc.pack(anchor='w', pady=4)
        f2_badge = tk.Label(card2, text="IN DEVELOPMENT", bg="#374151", fg="#9ca3af", font=('Segoe UI', 8, 'bold'), padx=6, pady=2)
        f2_badge.pack(anchor='e', pady=(4, 0))
        
        # Facility Card 3 (LOCKED)
        card3 = tk.Frame(card_container, bg="#111827", highlightbackground="#374151", highlightthickness=1, padx=12, pady=10)
        card3.pack(side='left', expand=True, fill='both', padx=5)
        
        f3_title = tk.Label(card3, text="⚡ Facility 3: Gravity Shift Void", bg="#111827", fg="#6b7280", font=('Segoe UI', 11, 'bold'))
        f3_title.pack(anchor='w')
        f3_desc = tk.Label(
            card3,
            text="• Variable Inverted Gravity Vectors\n• Floating Zero-G Platforms\n• Kinetic Trampoline Launchers\n• Extreme Physics Testing",
            bg="#111827", fg="#4b5563", font=('Segoe UI', 9), justify='left'
        )
        f3_desc.pack(anchor='w', pady=4)
        f3_badge = tk.Label(card3, text="LOCKED", bg="#374151", fg="#9ca3af", font=('Segoe UI', 8, 'bold'), padx=6, pady=2)
        f3_badge.pack(anchor='e', pady=(4, 0))

    def _build_dashboard_controls(self):
        sec_lbl = tk.Label(self.root, text="LAUNCH CONTROL & DASHBOARD ACTIONS", bg="#0c1017", fg="#00f0ff", font=('Segoe UI', 11, 'bold'))
        sec_lbl.pack(anchor='w', padx=18, pady=(12, 4))
        
        btn_container = tk.Frame(self.root, bg="#0c1017")
        btn_container.pack(fill='x', padx=15, pady=5)
        
        # Button 1: Start Visual Training
        btn1 = tk.Button(
            btn_container, text="🟢 Start Visual Training (GUI)", bg="#10b981", fg="#ffffff",
            font=('Segoe UI', 11, 'bold'), relief='flat', padx=12, pady=8, cursor='hand2',
            command=self.cmd_start_visual_training
        )
        btn1.pack(side='left', expand=True, fill='x', padx=4)
        
        # Button 2: Accelerated Training (Headless)
        btn2 = tk.Button(
            btn_container, text="⚡ Accelerated Training (Headless)", bg="#00a8ff", fg="#ffffff",
            font=('Segoe UI', 11, 'bold'), relief='flat', padx=12, pady=8, cursor='hand2',
            command=self.cmd_start_headless_training
        )
        btn2.pack(side='left', expand=True, fill='x', padx=4)
        
        # Button 3: Watch Bob Test Run
        btn3 = tk.Button(
            btn_container, text="👁️ Watch Bob Test Run (Demo)", bg="#8b5cf6", fg="#ffffff",
            font=('Segoe UI', 11, 'bold'), relief='flat', padx=12, pady=8, cursor='hand2',
            command=self.cmd_watch_demo
        )
        btn3.pack(side='left', expand=True, fill='x', padx=4)
        
        # Row 2 Actions
        btn_container2 = tk.Frame(self.root, bg="#0c1017")
        btn_container2.pack(fill='x', padx=15, pady=5)
        
        btn4 = tk.Button(
            btn_container2, text="📊 Room Statistics", bg="#1e293b", fg="#e2e8f0",
            font=('Segoe UI', 10, 'bold'), relief='flat', padx=10, pady=6, cursor='hand2',
            command=self.cmd_view_stats
        )
        btn4.pack(side='left', expand=True, fill='x', padx=4)
        
        btn5 = tk.Button(
            btn_container2, text="💾 Save Checkpoint", bg="#1e293b", fg="#e2e8f0",
            font=('Segoe UI', 10, 'bold'), relief='flat', padx=10, pady=6, cursor='hand2',
            command=self.cmd_save_checkpoint
        )
        btn5.pack(side='left', expand=True, fill='x', padx=4)
        
        btn6 = tk.Button(
            btn_container2, text="📂 Load Checkpoint", bg="#1e293b", fg="#e2e8f0",
            font=('Segoe UI', 10, 'bold'), relief='flat', padx=10, pady=6, cursor='hand2',
            command=self.cmd_load_checkpoint
        )
        btn6.pack(side='left', expand=True, fill='x', padx=4)
        
        btn7 = tk.Button(
            btn_container2, text="🔄 Reset Model Weights (Dumb Start)", bg="#ef4444", fg="#ffffff",
            font=('Segoe UI', 10, 'bold'), relief='flat', padx=10, pady=6, cursor='hand2',
            command=self.cmd_reset_weights
        )
        btn7.pack(side='left', expand=True, fill='x', padx=4)

    def _build_telemetry_panel(self):
        sec_lbl = tk.Label(self.root, text="STAGE CONFIGURATION & REAL-TIME TELEMETRY", bg="#0c1017", fg="#00f0ff", font=('Segoe UI', 11, 'bold'))
        sec_lbl.pack(anchor='w', padx=18, pady=(12, 4))
        
        telem_frame = tk.Frame(self.root, bg="#161d2a", padx=15, pady=12)
        telem_frame.pack(fill='x', padx=15, pady=5)
        
        # Stage Level Selector Spinbox
        stage_lbl = tk.Label(telem_frame, text="Current Stage Level:", bg="#161d2a", fg="#ffffff", font=('Segoe UI', 10, 'bold'))
        stage_lbl.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.stage_spin = ttk.Spinbox(telem_frame, from_=1, to=20, width=8, command=self._on_stage_changed)
        self.stage_spin.set(1)
        self.stage_spin.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Telemetry Metrics Display
        self.lbl_eps = tk.Label(telem_frame, text=f"Epsilon (Exploration): {self.agent.epsilon:.3f}", bg="#161d2a", fg="#38bdf8", font=('Segoe UI', 10, 'bold'))
        self.lbl_eps.grid(row=0, column=2, sticky='w', padx=20, pady=5)
        
        self.lbl_buffer = tk.Label(telem_frame, text=f"Replay Buffer: {len(self.agent.memory)} / {config.BUFFER_SIZE}", bg="#161d2a", fg="#38bdf8", font=('Segoe UI', 10, 'bold'))
        self.lbl_buffer.grid(row=0, column=3, sticky='w', padx=20, pady=5)
        
        # Status Log Output Textbox
        self.log_txt = tk.Text(self.root, height=5, bg="#090d14", fg="#a7f3d0", font=('Consolas', 9), relief='flat', padx=10, pady=8)
        self.log_txt.pack(fill='both', expand=True, padx=15, pady=8)
        self.log_msg("Ready. Select an action above to launch Bob's RL Testing Facility.")

    def _build_footer(self):
        footer_frame = tk.Frame(self.root, bg="#070a0f", padx=15, pady=6)
        footer_frame.pack(fill='x', side='bottom')
        
        git_lbl = tk.Label(
            footer_frame,
            text="Git Sync: ENABLED  |  Repository: https://github.com/sakrit204-byte/bob_reinforcement",
            bg="#070a0f", fg="#64748b", font=('Segoe UI', 9)
        )
        git_lbl.pack(side='left')
        
        ver_lbl = tk.Label(footer_frame, text="Bob RL Engine v2.0 (PyBullet 3D + PyTorch)", bg="#070a0f", fg="#64748b", font=('Segoe UI', 9))
        ver_lbl.pack(side='right')

    def log_msg(self, msg):
        self.log_txt.insert('end', f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_txt.see('end')
        
    def _on_stage_changed(self):
        try:
            self.selected_level = int(self.stage_spin.get())
            self.log_msg(f"Stage Level set to {self.selected_level:02d}")
        except:
            pass

    def cmd_start_visual_training(self):
        import subprocess
        level = int(self.stage_spin.get())
        self.log_msg(f"Launching Visual Training Room (GUI Window) for Stage {level:02d} in dedicated process...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bob.py")
        subprocess.Popen([sys.executable, script_path, "--visual", "--level", str(level)])

    def cmd_start_headless_training(self):
        import subprocess
        level = int(self.stage_spin.get())
        self.log_msg(f"Launching High-Speed Accelerated Headless Training for Stage {level:02d}...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bob.py")
        subprocess.Popen([sys.executable, script_path, "--headless", "--level", str(level)])

    def cmd_watch_demo(self):
        import subprocess
        level = int(self.stage_spin.get())
        self.log_msg(f"Launching Watch Bob Test Run (Demo Mode) for Stage {level:02d}...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bob.py")
        subprocess.Popen([sys.executable, script_path, "--demo", "--level", str(level)])

    def cmd_view_stats(self):
        times = self.agent.time_manager.best_times
        stats_str = "\n".join([f"Stage {lvl:02d}: {t:.2f}s" for lvl, t in sorted(times.items())])
        messagebox.showinfo("Facility Statistics & Best Completion Times", stats_str if stats_str else "No stage completion records yet.")

    def cmd_save_checkpoint(self):
        filename = f"checkpoint_stage_{self.selected_level:02d}.pth"
        self.agent.save(filename)
        self.log_msg(f"Saved Checkpoint successfully: bob_saves/{filename}")

    def cmd_load_checkpoint(self):
        filename = f"checkpoint_stage_{self.selected_level:02d}.pth"
        path = os.path.join(config.SAVE_DIR, filename)
        if os.path.exists(path):
            self.agent.load(filename)
            self.log_msg(f"Loaded Checkpoint successfully from {path}")
        else:
            messagebox.showwarning("Load Checkpoint", f"No checkpoint file found at {path}")

    def cmd_reset_weights(self):
        if messagebox.askyesno("Reset Model Weights", "Wipe model weights back to PURE DUMB RANDOM initialization?"):
            self.agent.reset_weights()
            self.log_msg("Model weights reset to PURE DUMB RANDOM initialization!")
            self._update_telemetry_labels()

    def _update_telemetry_labels(self):
        self.lbl_eps.config(text=f"Epsilon (Exploration): {self.agent.epsilon:.3f}")
        self.lbl_buffer.config(text=f"Replay Buffer: {len(self.agent.memory)} / {config.BUFFER_SIZE}")

def main():
    root = tk.Tk()
    app = BobsFacilityApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
