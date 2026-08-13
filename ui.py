"""
UI Scoreboard HUD texture generators for Bob's World 3D.
Uses PIL to draw real-time telemetry boards mapped onto PyBullet visual shapes.

Two billboards:
  1. Stats Board (back wall) - Stage, plates, neural model status
  2. Door Status Board (right side wall near exit) - Red/Green dot + countdown timer
"""

import os
from PIL import Image, ImageDraw, ImageFont


class DigitalBoardScreen:
    """Generates the main stats billboard texture (back wall). Wide 4:1 landscape."""
    def __init__(self, save_path="C:/Users/ACER/Desktop/24bce2954/bob_saves/board_screen.png"):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.width = 512
        self.height = 128
        self.last_hash = ""
        self.cached_tex_id = None

    def generate(self, stage_num=1, remaining_time=18.0, active_plates=0, total_plates=1, door_status="DOOR LOCKED"):
        """Returns (save_path, changed) where changed is True if the image was regenerated."""
        # Round time to whole seconds so texture only reloads once/second (not 10x/sec)
        state_hash = f"{stage_num}_{int(remaining_time)}_{active_plates}_{total_plates}_{door_status}"
        if state_hash == self.last_hash and os.path.exists(self.save_path):
            return self.save_path, False

        self.last_hash = state_hash
        W, H = self.width, self.height
        img = Image.new("RGBA", (W, H), (8, 12, 20, 230))
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("arialbd.ttf", 14)
            font_val = ImageFont.truetype("arialbd.ttf", 18)
            font_sm = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font_title = ImageFont.load_default()
            font_val = font_title
            font_sm = font_title

        # Thin cyan frame
        draw.rectangle([1, 1, W - 2, H - 2], outline=(0, 200, 240, 200), width=2)

        # Header line
        draw.text((10, 6), f"STAGE {stage_num:02d}", fill=(0, 235, 255), font=font_title)
        draw.line([10, 26, W - 10, 26], fill=(0, 200, 240, 80), width=1)

        # Three columns: TIME | PLATES | DOOR
        col_w = W // 3

        # Col 1: Time
        time_color = (50, 255, 120) if remaining_time > 10.0 else ((255, 200, 30) if remaining_time > 5.0 else (255, 50, 50))
        draw.text((12, 33), "TIME", fill=(100, 140, 160), font=font_sm)
        draw.text((12, 48), f"{int(remaining_time)}s", fill=time_color, font=font_val)

        # Col 2: Plates
        plates_color = (50, 255, 120) if active_plates == total_plates else (255, 200, 40)
        draw.text((col_w + 12, 33), "PLATES", fill=(100, 140, 160), font=font_sm)
        draw.text((col_w + 12, 48), f"{active_plates}/{total_plates}", fill=plates_color, font=font_val)

        # Col 3: Door
        door_open = door_status == "DOOR OPEN"
        door_color = (50, 255, 120) if door_open else (255, 60, 60)
        draw.text((col_w * 2 + 12, 33), "EXIT", fill=(100, 140, 160), font=font_sm)
        draw.text((col_w * 2 + 12, 48), "OPEN" if door_open else "LOCKED", fill=door_color, font=font_val)

        # Plate progress bar
        draw.line([10, 78, W - 10, 78], fill=(0, 200, 240, 60), width=1)
        bar_x, bar_y, bar_w, bar_h = 10, 84, W - 20, 10
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(0, 140, 180, 100), width=1)
        if total_plates > 0:
            fill_w = int(bar_w * active_plates / total_plates)
            if fill_w > 0:
                draw.rectangle([bar_x + 1, bar_y + 1, bar_x + fill_w, bar_y + bar_h - 1], fill=(50, 255, 120, 180))

        # Footer
        draw.text((10, 102), "PRESS 'N' FOR NEURAL NETWORK", fill=(0, 160, 190, 150), font=font_sm)

        img.save(self.save_path)
        return self.save_path, True


class DoorStatusScreen:
    """Generates door status indicator (right wall near exit). Landscape 2:1 layout.
    Left side: red/green dot. Right side: countdown timer."""
    def __init__(self, save_path="C:/Users/ACER/Desktop/24bce2954/bob_saves/door_status.png"):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.width = 256
        self.height = 128
        self.last_hash = ""
        self.cached_tex_id = None

    def generate(self, door_open=False, remaining_time=18.0):
        """Returns (save_path, changed)."""
        # Round time to whole seconds to prevent constant texture reload flashing
        state_hash = f"{'O' if door_open else 'C'}_{int(remaining_time)}"
        if state_hash == self.last_hash and os.path.exists(self.save_path):
            return self.save_path, False

        self.last_hash = state_hash
        W, H = self.width, self.height
        img = Image.new("RGBA", (W, H), (8, 12, 20, 220))
        draw = ImageDraw.Draw(img)

        try:
            font_time = ImageFont.truetype("arialbd.ttf", 28)
            font_label = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font_time = ImageFont.load_default()
            font_label = font_time

        # Thin frame
        draw.rectangle([1, 1, W - 2, H - 2], outline=(0, 200, 240, 180), width=2)

        # --- Left half: Status dot ---
        dot_cx, dot_cy = W // 4, H // 2
        dot_r = 28
        # Glow
        glow_color = (40, 255, 100, 60) if door_open else (255, 50, 50, 60)
        draw.ellipse([dot_cx - dot_r - 5, dot_cy - dot_r - 5,
                       dot_cx + dot_r + 5, dot_cy + dot_r + 5], fill=glow_color)
        # Main dot
        dot_fill = (40, 255, 100) if door_open else (255, 40, 40)
        dot_outline = (80, 255, 160) if door_open else (255, 100, 100)
        draw.ellipse([dot_cx - dot_r, dot_cy - dot_r,
                       dot_cx + dot_r, dot_cy + dot_r],
                      fill=dot_fill, outline=dot_outline, width=2)
        # Core
        core_r = 12
        core_fill = (180, 255, 220) if door_open else (255, 180, 180)
        draw.ellipse([dot_cx - core_r, dot_cy - core_r,
                       dot_cx + core_r, dot_cy + core_r], fill=core_fill)
        # Label
        label = "OPEN" if door_open else "LOCKED"
        label_color = (40, 255, 100) if door_open else (255, 60, 60)
        bbox = draw.textbbox((0, 0), label, font=font_label)
        lw = bbox[2] - bbox[0]
        draw.text((dot_cx - lw // 2, dot_cy + dot_r + 4), label, fill=label_color, font=font_label)

        # --- Right half: Timer ---
        right_cx = W * 3 // 4
        time_color = (50, 255, 120) if remaining_time > 10.0 else ((255, 200, 30) if remaining_time > 5.0 else (255, 50, 50))
        time_str = f"{int(remaining_time)}s"
        bbox = draw.textbbox((0, 0), time_str, font=font_time)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((right_cx - tw // 2, H // 2 - th // 2 - 6), time_str, fill=time_color, font=font_time)
        # Small label
        bbox = draw.textbbox((0, 0), "TIME", font=font_label)
        lw = bbox[2] - bbox[0]
        draw.text((right_cx - lw // 2, H // 2 + th // 2 + 2), "TIME", fill=(100, 140, 160), font=font_label)

        # Vertical divider between halves
        draw.line([W // 2, 8, W // 2, H - 8], fill=(0, 200, 240, 60), width=1)

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
        except Exception:
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

        img.save(self.save_path)
        return self.save_path, True
