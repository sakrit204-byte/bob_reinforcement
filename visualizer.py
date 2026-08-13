"""
2D Neural Network HUD Window for Bob's RL Agent.
Spawns as a separate always-on-top Tkinter window alongside the 3D PyBullet environment.
Renders live neuron nodes, synapse lines, and Q-value bars on a Canvas at 20 FPS.
"""

import threading
import tkinter as tk
import numpy as np


class NeuralNetworkHUD:
    """Lightweight 2D HUD window that renders the live neural network graph."""

    def __init__(self, agent_ref):
        self.agent = agent_ref
        self.alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.root = tk.Tk()
        self.root.title("Bob's Neural Network  |  Live Activations")
        self.root.geometry("680x440+30+30")
        self.root.configure(bg="#080c14")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.canvas = tk.Canvas(self.root, bg="#080c14", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._draw()
        self.root.mainloop()

    def _on_close(self):
        self.alive = False
        self.root.destroy()

    def _draw(self):
        if not self.alive:
            return

        c = self.canvas
        c.delete("all")
        W = c.winfo_width() or 660
        H = c.winfo_height() or 420

        # Border
        c.create_rectangle(2, 2, W - 2, H - 2, outline="#00c8dc", width=2)

        # Title
        c.create_text(W // 2, 18, text="NEURAL NETWORK  |  LIVE ACTIVATIONS",
                       fill="#00e6ff", font=("Segoe UI", 12, "bold"))

        # Get activations
        acts = getattr(self.agent, "latest_activations", None) if self.agent else None
        inp = np.array(acts.get("input", np.zeros(16)), dtype=float) if acts else np.zeros(16)
        h1 = np.array(acts.get("h1", np.zeros(128)), dtype=float) if acts else np.zeros(128)
        h2 = np.array(acts.get("h2", np.zeros(128)), dtype=float) if acts else np.zeros(128)
        q_vals = np.array(acts.get("q_values", np.zeros(5)), dtype=float) if acts else np.zeros(5)
        best = int(np.argmax(q_vals))

        # Column X positions
        cx = [int(W * 0.10), int(W * 0.35), int(W * 0.60), int(W * 0.86)]

        # Layer headers
        headers = ["INPUT (16)", "HIDDEN L1 (128)", "HIDDEN L2 (128)", "OUTPUT (5)"]
        for i, h in enumerate(headers):
            c.create_text(cx[i], 42, text=h, fill="#5a9ab5", font=("Segoe UI", 8))

        # Build node lists  [  (x, y, value, label)  ]
        in_labels = ["X", "Y", "VelX", "VelY", "dx", "dy", "Gnd", "Time"]
        act_labels = ["BACK", "FWD", "LEFT", "RIGHT", "JUMP"]
        nodes = [[], [], [], []]

        for i in range(8):
            y = 68 + i * 40
            v = abs(float(inp[i])) if i < len(inp) else 0.0
            nodes[0].append((cx[0], y, v, in_labels[i]))

        for i in range(8):
            y = 68 + i * 40
            v = float(h1[i * 16]) if (i * 16) < len(h1) else 0.0
            nodes[1].append((cx[1], y, v, ""))

        for i in range(8):
            y = 68 + i * 40
            v = float(h2[i * 16]) if (i * 16) < len(h2) else 0.0
            nodes[2].append((cx[2], y, v, ""))

        for i in range(5):
            y = 100 + i * 60
            v = float(q_vals[i])
            nodes[3].append((cx[3], y, v, act_labels[i]))

        # Synapse lines
        for layer in range(3):
            for (x1, y1, v1, _) in nodes[layer]:
                for (x2, y2, v2, _) in nodes[layer + 1]:
                    intensity = min(1.0, max(0.0, (abs(v1) + abs(v2)) * 0.35))
                    if intensity < 0.04:
                        continue
                    g = int(intensity * 160)
                    b = int(100 + intensity * 155)
                    color = f"#{int(intensity*30):02x}{g:02x}{b:02x}"
                    c.create_line(x1, y1, x2, y2, fill=color, width=1)

        # Nodes
        R = 11
        for layer in range(4):
            for i, (nx, ny, nv, nl) in enumerate(nodes[layer]):
                if layer == 3:
                    is_best = (i == best)
                    fill = "#0ac86a" if is_best else "#1e2838"
                    outline = "#00ffb4" if is_best else "#4a5568"
                    c.create_oval(nx - R, ny - R, nx + R, ny + R,
                                  fill=fill, outline=outline, width=3 if is_best else 1)
                    tc = "#ffffff" if is_best else "#8899aa"
                    prefix = ">> " if is_best else ""
                    c.create_text(nx + R + 6, ny - 7, anchor="w",
                                  text=f"{prefix}{nl}", fill=tc,
                                  font=("Segoe UI", 10, "bold" if is_best else "normal"))
                    c.create_text(nx + R + 6, ny + 8, anchor="w",
                                  text=f"Q: {nv:+.2f}", fill=tc,
                                  font=("Segoe UI", 8))
                else:
                    active = abs(nv) > 0.1
                    gv = int(min(255, abs(nv) * 350))
                    bv = int(min(255, 100 + abs(nv) * 280))
                    fill = f"#00{gv:02x}{bv:02x}" if active else "#141c28"
                    outline = "#00c8f0" if active else "#2d3748"
                    c.create_oval(nx - R, ny - R, nx + R, ny + R,
                                  fill=fill, outline=outline, width=2 if active else 1)
                    if nl:
                        c.create_text(nx - R - 4, ny, anchor="e",
                                      text=nl, fill="#7a8ea0", font=("Segoe UI", 8))

        # Bottom bar
        c.create_text(W // 2, H - 14,
                       text=f"WINNING: {act_labels[best]}  |  Q: {q_vals[best]:+.2f}  |  Close window or press 'N' to hide",
                       fill="#00d0e8", font=("Segoe UI", 9))

        # Schedule next frame (20 FPS)
        if self.alive:
            self.root.after(50, self._draw)

    def close(self):
        self.alive = False
        try:
            self.root.after(0, self.root.destroy)
        except:
            pass
