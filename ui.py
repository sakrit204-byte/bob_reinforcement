"""
UI Scoreboard HUD texture generators for Bob's World 3D.
Uses PIL to draw real-time telemetry boards mapped onto PyBullet visual shapes.
"""

import os
from PIL import Image, ImageDraw, ImageFont


class DigitalBoardScreen:
    """Generates a real 3D texture image for the obsidian blackboard mesh surface."""
    def __init__(self, save_path="C:/Users/ACER/Desktop/24bce2954/bob_saves/board_screen.png"):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.width = 1024
        self.height = 256
        self.last_hash = ""
        self.cached_tex_id = None  # Track loaded texture to avoid leaks
        
    def generate(self, stage_num=1, remaining_time=18.0, active_plates=0, total_plates=1, door_status="DOOR LOCKED"):
        """Returns (save_path, changed) where changed is True if the image was regenerated."""
        state_hash = f"{stage_num}_{remaining_time:.1f}_{active_plates}_{total_plates}_{door_status}"
        if state_hash == self.last_hash and os.path.exists(self.save_path):
            return self.save_path, False
            
        self.last_hash = state_hash
        img = Image.new("RGBA", (self.width, self.height), (8, 12, 20, 255))
        draw = ImageDraw.Draw(img)
        
        # Outer Glowing LED Frame Borders
        draw.rectangle([4, 4, self.width - 5, self.height - 5], outline=(0, 240, 255, 255), width=6)
        draw.rectangle([10, 10, self.width - 11, self.height - 11], outline=(0, 180, 220, 180), width=2)
        
        try:
            font_title = ImageFont.truetype("arialbd.ttf", 34)
            font_large = ImageFont.truetype("arialbd.ttf", 36)
            font_small = ImageFont.truetype("arial.ttf", 26)
        except:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            
        # Line 1: Facility Title
        header_text = f"[*] BOB'S TESTING FACILITY  |  STAGE {stage_num:02d}"
        draw.text((25, 20), header_text, fill=(0, 240, 255, 255), font=font_title)
        
        # Divider Line
        draw.line([25, 75, self.width - 25, 75], fill=(0, 240, 255, 120), width=2)
        
        # Line 2: Telemetry Metrics - Time, Plates, Door
        time_color = (50, 255, 120, 255) if remaining_time > 10.0 else ((255, 200, 30, 255) if remaining_time > 5.0 else (255, 50, 50, 255))
        draw.text((25, 95), f"TIME: {remaining_time:.1f}s", fill=time_color, font=font_large)
        
        plates_color = (50, 255, 120, 255) if active_plates == total_plates else (255, 220, 50, 255)
        draw.text((340, 95), f"PLATES: {active_plates}/{total_plates} GREEN", fill=plates_color, font=font_large)
        
        door_color = (50, 255, 120, 255) if door_status == "DOOR OPEN" else (255, 60, 60, 255)
        draw.text((720, 95), f"{door_status}", fill=door_color, font=font_large)
        
        # Line 3: Bottom Status Bar
        draw.line([25, 160, self.width - 25, 160], fill=(0, 240, 255, 120), width=2)
        draw.text((25, 185), "NEURAL MODEL: ACTIVE  |  PRESS 'N' FOR LIVE NETWORK GRAPH", fill=(0, 220, 255, 255), font=font_small)
        
        # Horizontal flip for correct PyBullet box texture mapping (cancels out UV horizontal inversion)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(self.save_path)
        return self.save_path, True


class NeuralNetworkScreen:
    """Generates a neural network node-and-synapse diagram as a PIL texture (kept for 3D mapping compatibility)."""
    def __init__(self, save_path="C:/Users/ACER/Desktop/24bce2954/bob_saves/nn_screen.png"):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.width = 640
        self.height = 400
        self.last_hash = ""
        
        try:
            self.font_title = ImageFont.truetype("arialbd.ttf", 16)
            self.font_label = ImageFont.truetype("arial.ttf", 11)
            self.font_val = ImageFont.truetype("arialbd.ttf", 13)
        except:
            self.font_title = ImageFont.load_default()
            self.font_label = ImageFont.load_default()
            self.font_val = ImageFont.load_default()

    def generate(self, inp, h1, h2, q_vals, best_act):
        """Returns (save_path, changed)."""
        import numpy as np
        sig = f"{best_act}_{np.round(q_vals, 1).tobytes().hex()[:16]}"
        if sig == self.last_hash and os.path.exists(self.save_path):
            return self.save_path, False
        self.last_hash = sig

        W, H = self.width, self.height
        img = Image.new("RGB", (W, H), (10, 14, 22))
        draw = ImageDraw.Draw(img)

        # Border
        draw.rectangle([2, 2, W - 3, H - 3], outline=(0, 200, 220), width=2)

        # Title
        draw.text((W // 2 - 120, 8), "NEURAL NETWORK  |  LIVE ACTIVATIONS", fill=(0, 230, 255), font=self.font_title)

        # Layer columns
        col_x = [int(W * 0.10), int(W * 0.35), int(W * 0.60), int(W * 0.85)]
        layer_labels = ["INPUT (16)", "HIDDEN L1 (128)", "HIDDEN L2 (128)", "OUTPUT (5)"]
        for i, lbl in enumerate(layer_labels):
            draw.text((col_x[i] - 30, 32), lbl, fill=(100, 180, 220), font=self.font_label)

        input_names = ["X", "Y", "VX", "VY", "dx", "dy", "Gnd", "T"]
        action_names = ["BACK", "FWD", "LEFT", "RIGHT", "JUMP"]

        nodes = [[], [], [], []]
        for i in range(8):
            y = 60 + i * 38
            val = abs(float(inp[i])) if i < len(inp) else 0.0
            nodes[0].append((col_x[0], y, val, input_names[i]))
        for i in range(8):
            y = 60 + i * 38
            val = float(h1[i * 16]) if (i * 16) < len(h1) else 0.0
            nodes[1].append((col_x[1], y, val, ""))
        for i in range(8):
            y = 60 + i * 38
            val = float(h2[i * 16]) if (i * 16) < len(h2) else 0.0
            nodes[2].append((col_x[2], y, val, ""))
        for i in range(5):
            y = 90 + i * 55
            val = float(q_vals[i])
            nodes[3].append((col_x[3], y, val, action_names[i]))

        for l in range(3):
            for (x1, y1, v1, _) in nodes[l]:
                for (x2, y2, v2, _) in nodes[l + 1]:
                    intensity = min(1.0, max(0.0, (abs(v1) + abs(v2)) * 0.4))
                    if intensity < 0.05:
                        continue
                    r = int(intensity * 40)
                    g = int(intensity * 180)
                    b = int(120 + intensity * 135)
                    draw.line([(x1, y1), (x2, y2)], fill=(r, g, b), width=1)

        R = 10
        for l in range(4):
            for i, (nx, ny, nval, nlabel) in enumerate(nodes[l]):
                if l == 3:  # Output
                    is_best = (i == best_act)
                    fill = (10, 200, 100) if is_best else (30, 40, 55)
                    outline = (0, 255, 180) if is_best else (80, 100, 120)
                    draw.ellipse([nx - R, ny - R, nx + R, ny + R], fill=fill, outline=outline, width=2)
                    tc = (255, 255, 255) if is_best else (160, 170, 180)
                    prefix = ">> " if is_best else ""
                    draw.text((nx + R + 4, ny - 8), f"{prefix}{nlabel}", fill=tc, font=self.font_val)
                    draw.text((nx + R + 4, ny + 4), f"Q:{nval:+.1f}", fill=tc, font=self.font_label)
                else:
                    active = abs(nval) > 0.1
                    fill = (0, int(min(255, abs(nval) * 400)), int(min(255, 120 + abs(nval) * 300))) if active else (20, 28, 38)
                    outline = (0, 200, 240) if active else (50, 60, 75)
                    draw.ellipse([nx - R, ny - R, nx + R, ny + R], fill=fill, outline=outline, width=2 if active else 1)
                    if nlabel:
                        draw.text((nx - R - 22, ny - 6), nlabel, fill=(140, 160, 180), font=self.font_label)

        draw.text((10, H - 22), f"WINNING ACTION: {action_names[best_act]}  |  Q: {q_vals[best_act]:+.2f}  |  PRESS 'N' TO HIDE", fill=(0, 220, 240), font=self.font_label)

        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(self.save_path)
        return self.save_path, True
