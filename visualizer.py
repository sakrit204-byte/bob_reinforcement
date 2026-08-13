"""
Interactive 2D Neural Network Visualizer Window for Bob's World RL Agent.
Can be moved anywhere on screen, resized, and interacted with in real-time alongside the 3D PyBullet Environment.
Renders real-time neuron nodes, synapse connection lines, and action Q-value charts.
"""

import tkinter as tk
import math
import numpy as np

class InteractiveNeuralVisualizer:
    """
    Dedicated 2D Window Overlay that renders live PyTorch neuron activations,
    synapse connections, and action Q-values.
    """
    def __init__(self, master=None, agent=None):
        if master is None:
            self.root = tk.Tk()
            self.is_toplevel = False
        else:
            self.root = tk.Toplevel(master)
            self.is_toplevel = True
            
        self.root.title("🧠 Bob's Neural Brain Visualizer (Live PyTorch Model)")
        self.root.geometry("850x520+50+50")
        self.root.configure(bg="#0c1017")
        self.agent = agent
        self.running = True
        
        # Title Bar
        header = tk.Frame(self.root, bg="#161d2a", padx=10, pady=8)
        header.pack(fill='x')
        lbl_title = tk.Label(header, text="🧠 BOB'S DEDICATED LIVE NEURAL NETWORK VISUALIZER", bg="#161d2a", fg="#00f0ff", font=('Segoe UI', 12, 'bold'))
        lbl_title.pack(side='left')
        lbl_info = tk.Label(header, text="Drag window to reposition | Live PyTorch Activations", bg="#161d2a", fg="#94a3b8", font=('Segoe UI', 9))
        lbl_info.pack(side='right')
        
        # Interactive Canvas
        self.canvas = tk.Canvas(self.root, bg="#090d14", highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Protocol on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initial draw loop
        self.draw_network()
        
    def on_close(self):
        self.running = False
        self.root.destroy()
        
    def draw_network(self):
        if not self.running:
            return
            
        try:
            self.canvas.delete("all")
            width = self.canvas.winfo_width() or 830
            height = self.canvas.winfo_height() or 450
            
            acts = getattr(self.agent, 'latest_activations', None) if self.agent else None
            inp = acts.get('input', np.zeros(16)) if acts else np.zeros(16)
            h1 = acts.get('h1', np.zeros(64)) if acts else np.zeros(64)
            h2 = acts.get('h2', np.zeros(64)) if acts else np.zeros(64)
            q_vals = acts.get('q_values', np.zeros(5)) if acts else np.zeros(5)
            best_act = np.argmax(q_vals)
            
            # Layer X Coordinates
            col_x = [width * 0.14, width * 0.38, width * 0.62, width * 0.85]
            
            # 1. Draw Layer Labels
            self.canvas.create_text(col_x[0], 25, text="INPUT LAYER (16)", fill="#00f0ff", font=('Segoe UI', 10, 'bold'))
            self.canvas.create_text(col_x[1], 25, text="HIDDEN L1 (64)", fill="#38bdf8", font=('Segoe UI', 10, 'bold'))
            self.canvas.create_text(col_x[2], 25, text="HIDDEN L2 (64)", fill="#38bdf8", font=('Segoe UI', 10, 'bold'))
            self.canvas.create_text(col_x[3], 25, text="OUTPUT ACTIONS (5)", fill="#a7f3d0", font=('Segoe UI', 10, 'bold'))
            
            # 2. Compute Node Positions
            # Input Nodes (8 sampled)
            input_names = ["Bob X", "Bob Y", "Vel X", "Vel Y", "Target dx", "Target dy", "Grounded", "Time Rem"]
            in_nodes = []
            for i in range(8):
                y = 65 + i * 45
                in_nodes.append((col_x[0], y, abs(float(inp[i])) if i < len(inp) else 0.0, input_names[i]))
                
            # Hidden 1 Nodes (8 sampled)
            h1_nodes = []
            for i in range(8):
                y = 65 + i * 45
                val = float(h1[i * 8]) if (i * 8) < len(h1) else 0.0
                h1_nodes.append((col_x[1], y, val, f"H1-{i+1}"))
                
            # Hidden 2 Nodes (8 sampled)
            h2_nodes = []
            for i in range(8):
                y = 65 + i * 45
                val = float(h2[i * 8]) if (i * 8) < len(h2) else 0.0
                h2_nodes.append((col_x[2], y, val, f"H2-{i+1}"))
                
            # Output Action Nodes (5)
            action_names = ["0: BACK (-X)", "1: FWD (+X)", "2: LEFT (-Y)", "3: RIGHT (+Y)", "4: JUMP (+Z)"]
            out_nodes = []
            for i in range(5):
                y = 95 + i * 65
                out_nodes.append((col_x[3], y, float(q_vals[i]), action_names[i]))
                
            # 3. Draw Synapse Connection Lines (Edges) between layers
            layers = [in_nodes, h1_nodes, h2_nodes, out_nodes]
            for l in range(3):
                curr_l = layers[l]
                next_l = layers[l+1]
                for n1 in curr_l[:4]:
                    for n2 in next_l[:4]:
                        val1, val2 = n1[2], n2[2]
                        intensity = min(1.0, max(0.1, (val1 + val2) * 0.5))
                        color = f"#{int(intensity*0):02x}{int(intensity*200):02x}{int(intensity*255):02x}"
                        width_line = 1 if intensity < 0.4 else 2
                        self.canvas.create_line(n1[0] + 15, n1[1], n2[0] - 15, n2[1], fill=color, width=width_line)

            # 4. Draw Nodes (Circles & Labels)
            for l_idx, layer in enumerate(layers):
                for i, (nx, ny, nval, nlabel) in enumerate(layer):
                    r = 14
                    if l_idx == 3: # Output layer
                        is_winner = (i == best_act)
                        fill_col = "#059669" if is_winner else "#1e293b"
                        outline_col = "#00f0ff" if is_winner else "#475569"
                        text_col = "#ffffff" if is_winner else "#94a3b8"
                        
                        # Node Circle
                        self.canvas.create_oval(nx - r, ny - r, nx + r, ny + r, fill=fill_col, outline=outline_col, width=3 if is_winner else 1)
                        # Action Label + Q-Value
                        pref = "★ WINNER: " if is_winner else ""
                        self.canvas.create_text(nx + 25, ny, text=f"{pref}{nlabel} [Q: {nval:+.2f}]", fill=text_col, font=('Segoe UI', 9, 'bold' if is_winner else 'normal'), anchor='w')
                    else:
                        is_active = (nval > 0.1)
                        fill_col = "#00f0ff" if is_active else "#1e293b"
                        outline_col = "#38bdf8" if is_active else "#334155"
                        
                        self.canvas.create_oval(nx - r, ny - r, nx + r, ny + r, fill=fill_col, outline=outline_col, width=2 if is_active else 1)
                        self.canvas.create_text(nx - 22, ny, text=nlabel, fill="#94a3b8", font=('Segoe UI', 8), anchor='e')

        except Exception as e:
            pass
            
        # Schedule next update (20 FPS)
        if self.running:
            self.root.after(50, self.draw_network)
            
    def update_data(self, agent):
        self.agent = agent

def launch_standalone_visualizer(agent=None):
    vis = InteractiveNeuralVisualizer(master=None, agent=agent)
    vis.root.mainloop()
