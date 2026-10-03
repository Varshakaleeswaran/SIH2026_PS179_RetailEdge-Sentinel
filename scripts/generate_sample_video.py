"""
RetailEdge Sentinel - Synthetic Retail CCTV Video Generator
Creates a realistic 640x480 retail surveillance video:
- Entrance area with entry gates
- Shelf aisle with merchandise displays
- Billing checkout counter
- Walking animated human figures demonstrating customer journey:
  Entrance -> Shelf Browsing -> Queue Buildup -> Checkout -> Exit
"""

import os
import math
import random
from pathlib import Path
import cv2
import numpy as np

def draw_person(frame: np.ndarray, x: int, y: int, height: int = 70, shirt_color=(60, 120, 200), pants_color=(50, 50, 60)):
    """Draw a realistic synthetic person figure on the retail floor frame."""
    head_r = int(height * 0.12)
    head_y = y - int(height * 0.85)
    
    # Shadow
    cv2.ellipse(frame, (x, y), (int(head_r * 1.8), int(head_r * 0.7)), 0, 0, 360, (150, 150, 150), -1)

    # Legs / Pants
    leg_w = int(head_r * 0.6)
    cv2.rectangle(frame, (x - leg_w - 1, head_y + head_r * 3), (x - 2, y), pants_color, -1)
    cv2.rectangle(frame, (x + 2, head_y + head_r * 3), (x + leg_w + 1, y), pants_color, -1)

    # Torso / Shirt
    torso_w = int(head_r * 1.9)
    torso_h = int(height * 0.45)
    cv2.rectangle(frame, (x - torso_w // 2, head_y + head_r + 2), (x + torso_w // 2, head_y + head_r + torso_h), shirt_color, -1)

    # Head
    cv2.circle(frame, (x, head_y), head_r, (180, 210, 235), -1)
    cv2.circle(frame, (x, head_y), head_r, (130, 160, 190), 1)

def draw_retail_background(width=640, height=480) -> np.ndarray:
    """Draw standard retail store interior floor, walls, shelf displays, and billing counter."""
    bg = np.full((height, width, 3), (225, 230, 235), dtype=np.uint8)

    # Floor tiles grid pattern
    tile_size = 40
    for y in range(0, height, tile_size):
        cv2.line(bg, (0, y), (width, y), (210, 215, 220), 1)
    for x in range(0, width, tile_size):
        cv2.line(bg, (x, 0), (x, height), (210, 215, 220), 1)

    # Entrance Door Gate (Top Left)
    cv2.rectangle(bg, (20, 20), (260, 40), (70, 80, 90), -1)
    cv2.putText(bg, "AUTOMATIC SLIDING DOORS", (35, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
    cv2.line(bg, (20, 190), (260, 190), (140, 150, 160), 2) # Entry line demarcator

    # Shelf Racks & Merchandise (Top Right)
    shelf_x1, shelf_y1, shelf_x2, shelf_y2 = 330, 40, 610, 220
    cv2.rectangle(bg, (shelf_x1, shelf_y1), (shelf_x2, shelf_y2), (180, 190, 195), -1)
    cv2.rectangle(bg, (shelf_x1, shelf_y1), (shelf_x2, shelf_y2), (110, 120, 130), 2)
    # Shelf rows with product packs
    row_colors = [(50, 180, 80), (200, 120, 40), (60, 70, 210), (180, 60, 180)]
    for idx, ry in enumerate(range(shelf_y1 + 25, shelf_y2 - 15, 40)):
        cv2.line(bg, (shelf_x1 + 10, ry), (shelf_x2 - 10, ry), (80, 90, 100), 2)
        # Draw simulated item boxes
        c = row_colors[idx % len(row_colors)]
        for bx in range(shelf_x1 + 15, shelf_x2 - 30, 22):
            cv2.rectangle(bg, (bx, ry - 18), (bx + 16, ry - 2), c, -1)
            cv2.rectangle(bg, (bx, ry - 18), (bx + 16, ry - 2), (40, 40, 40), 1)
    cv2.putText(bg, "AISLE 1: GROCERY & PACKAGED GOODS", (340, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1)

    # Billing Counter (Bottom Right)
    desk_x1, desk_y1, desk_x2, desk_y2 = 500, 300, 610, 440
    cv2.rectangle(bg, (desk_x1, desk_y1), (desk_x2, desk_y2), (140, 110, 90), -1)
    cv2.rectangle(bg, (desk_x1, desk_y1), (desk_x2, desk_y2), (70, 50, 40), 2)
    # POS Terminal & Cash Register
    cv2.rectangle(bg, (desk_x1 + 20, desk_y1 + 30), (desk_x1 + 65, desk_y1 + 75), (40, 40, 40), -1)
    cv2.rectangle(bg, (desk_x1 + 25, desk_y1 + 35), (desk_x1 + 60, desk_y1 + 65), (100, 220, 120), -1)
    cv2.putText(bg, "POS #1", (desk_x1 + 27, desk_y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (10, 10, 10), 1)
    cv2.putText(bg, "CHECKOUT DESK", (desk_x1 + 8, desk_y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # Cashier person sitting at desk
    draw_person(bg, desk_x1 + 80, desk_y1 + 80, height=65, shirt_color=(40, 140, 70), pants_color=(30, 30, 40))

    return bg

def generate_sample_cctv_video(output_path: str = "videos/sample_store.mp4", duration_sec: int = 15, fps: int = 25):
    """Generate a realistic retail surveillance video with walking customers."""
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    bg_base = draw_retail_background(w, h)

    # Define shopper customer agents with specific waypoints and timing
    # Customer cycle: Entrance -> Shelf Aisle -> Billing Queue -> Exit
    customers = [
        # Customer 1: starts near shelf, moves to queue
        {"t_start": 0, "path": [(400, 180), (450, 200), (400, 320), (460, 340), (480, 340)], "color": (190, 80, 50)},
        # Customer 2: enters through door, goes to shelf
        {"t_start": 10, "path": [(140, 60), (140, 170), (180, 220), (350, 160), (370, 160)], "color": (60, 140, 200)},
        # Customer 3: in queue waiting
        {"t_start": 0, "path": [(430, 340), (450, 340), (470, 340), (480, 340)], "color": (180, 120, 40)},
        # Customer 4: in queue waiting behind #3
        {"t_start": 0, "path": [(370, 340), (390, 340), (410, 340), (430, 340)], "color": (120, 60, 160)},
        # Customer 5: enters later and joins queue
        {"t_start": 60, "path": [(100, 60), (100, 180), (250, 290), (350, 340), (380, 340)], "color": (50, 160, 90)},
        # Customer 6: enters and browses shelf
        {"t_start": 120, "path": [(160, 60), (160, 190), (280, 210), (480, 170)], "color": (210, 90, 120)},
        # Customer 7: joins the growing queue
        {"t_start": 160, "path": [(120, 60), (130, 190), (220, 300), (320, 340), (340, 340)], "color": (80, 100, 190)},
    ]

    for frame_idx in range(total_frames):
        frame = bg_base.copy()

        # Update and render each customer
        for c in customers:
            if frame_idx < c["t_start"]:
                continue
            
            rel_f = frame_idx - c["t_start"]
            path = c["path"]
            total_steps = len(path) - 1
            segment_len = max(20, (total_frames - c["t_start"]) // total_steps)

            seg_idx = min(total_steps - 1, rel_f // segment_len)
            seg_progress = min(1.0, (rel_f % segment_len) / float(segment_len))

            p1 = path[seg_idx]
            p2 = path[seg_idx + 1]

            # Interpolate (x, y) with slight natural walking sway
            cur_x = int(p1[0] + (p2[0] - p1[0]) * seg_progress)
            cur_y = int(p1[1] + (p2[1] - p1[1]) * seg_progress)
            sway_y = int(math.sin(frame_idx * 0.4) * 2)

            draw_person(frame, cur_x, cur_y + sway_y, height=72, shirt_color=c["color"])

        # Timestamp overlay (CCTV format)
        sec = frame_idx / fps
        cctv_time = f"CAM-01 [MAIN FLOOR]  2026-09-30 14:35:{int(sec):02d}.{int((sec % 1) * 100):02d}"
        cv2.putText(frame, cctv_time, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"Sample retail CCTV video successfully created at {output_path} ({total_frames} frames)")

if __name__ == "__main__":
    generate_sample_cctv_video()
