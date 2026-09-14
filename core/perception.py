import cv2
import numpy as np
from ultralytics import YOLO

OBSTACLES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "dog", "cat", "horse",
    "sheep", "cow", "bear", "backpack", "suitcase", "bench", "chair", "fire hydrant",
    "stop sign", "traffic light"
}

class Perception:
    def __init__(self, det_weights="yolo11n.pt", seg_weights="yolo11n-seg.pt"):
        self.det = YOLO(det_weights)
        self.seg = YOLO(seg_weights)

    def run(self, frame, conf=0.30, enable_hud=True):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        obstacles = []

        # Object detection
        r = self.det.predict(frame, conf=conf, verbose=False)[0]
        if r.boxes is not None:
            names = r.names
            for b in r.boxes:
                cls = int(b.cls[0])
                label = names[cls]
                if label not in OBSTACLES:
                    continue
                
                score = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                box_h = max(1, y2 - y1)
                box_w = max(1, x2 - x1)
                area = (box_w * box_h) / (w * h + 1e-9)
                bottom = y2 / h
                center = 1.0 - abs(((x1 + x2) / 2.0) - w / 2.0) / (w / 2.0)

                # Distance proxy calculation (meters) using vertical perspective & box size
                norm_height = box_h / h
                distance_m = float(np.clip(1.8 / (bottom * 0.7 + norm_height * 0.3 + 1e-3), 0.8, 25.0))

                # Collision Risk Assessment
                risk = float(np.clip(0.25 * area * 18.0 + 0.45 * bottom + 0.30 * center, 0.0, 1.0))

                obstacles.append(dict(
                    label=label,
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    risk=risk,
                    distance_m=distance_m,
                    box_h=box_h,
                    box_w=box_w
                ))

                # Draw high-tech tactical bounding box
                if risk > 0.65:
                    col = (0, 0, 255)       # High Risk - Red
                elif risk > 0.35:
                    col = (0, 165, 255)     # Medium Risk - Orange
                else:
                    col = (255, 200, 0)     # Low Risk - Cyan/Yellow

                # Corner brackets instead of plain rectangle for military/UGV aesthetic
                corner_len = min(16, box_w // 4, box_h // 4)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 1)
                # Thick corners
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), col, 3)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), col, 3)
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), col, 3)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), col, 3)
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), col, 3)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), col, 3)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), col, 3)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), col, 3)

                # Tag Label Badge
                tag = f"{label.upper()} {score:.2f} | {distance_m:.1f}m | RISK:{risk:.2f}"
                (txt_w, txt_h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                badge_y1 = max(0, y1 - txt_h - 8)
                cv2.rectangle(annotated, (x1, badge_y1), (x1 + txt_w + 6, badge_y1 + txt_h + 6), col, -1)
                cv2.putText(annotated, tag, (x1 + 3, badge_y1 + txt_h + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Free space calculation & obstacle masking
        free = self.free_space(frame)
        for ob in obstacles:
            x1, y1, x2, y2 = ob["bbox"]
            pad = int(6 + 22 * ob["risk"])
            cv2.rectangle(free, (max(0, x1 - pad), max(0, y1 - pad)),
                          (min(w - 1, x2 + pad), min(h - 1, y2 + pad)), 0, -1)

        # Draw HUD overlays if enabled
        if enable_hud:
            annotated = self.draw_tactical_hud(annotated)

        return obstacles, free, annotated

    def free_space(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (20, 25, 20), (105, 255, 245))
        earth = cv2.inRange(hsv, (0, 15, 25), (38, 255, 235))
        road = cv2.inRange(hsv, (0, 0, 45), (180, 95, 235))
        mask = cv2.bitwise_or(green, cv2.bitwise_or(earth, road))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        texture = cv2.GaussianBlur(edges, (9, 9), 0)
        mask[texture > 125] = 0

        roi = np.zeros_like(mask)
        poly = np.array([[(int(.05 * w), int(.43 * h)), (int(.95 * w), int(.43 * h)),
                        (w - 1, h - 1), (0, h - 1)]])
        cv2.fillPoly(roi, poly, 255)
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        return mask

    def estimate_depth_map(self, frame):
        """Generates a monocular depth map proxy (Inferno colormap)."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Vertical gradient proxy (closer items at bottom are brighter/nearer)
        y_grid = np.linspace(0.1, 1.0, h).reshape(h, 1)
        depth_base = np.tile(y_grid, (1, w))
        
        # Blur & edge structure for object depth variations
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blur, 30, 100)
        edge_dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        cv2.normalize(edge_dist, edge_dist, 0, 1.0, cv2.NORM_MINMAX)
        
        depth_composite = np.clip(depth_base * 0.75 + (1.0 - edge_dist) * 0.25, 0, 1.0)
        depth_uint8 = (depth_composite * 255).astype(np.uint8)
        
        # Apply Inferno colormap for depth visualization
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        return depth_colored

    def draw_tactical_hud(self, frame):
        """Draws military/industrial Ground Control Station HUD elements."""
        h, w = frame.shape[:2]
        hud = frame.copy()
        
        # Center Targeting Reticle
        cx, cy = w // 2, h // 2
        reticle_col = (0, 255, 0)
        cv2.circle(hud, (cx, cy), 18, reticle_col, 1)
        cv2.circle(hud, (cx, cy), 3, reticle_col, -1)
        cv2.line(hud, (cx - 28, cy), (cx - 10, cy), reticle_col, 1)
        cv2.line(hud, (cx + 10, cy), (cx + 28, cy), reticle_col, 1)
        cv2.line(hud, (cx, cy - 28), (cx, cy - 10), reticle_col, 1)
        cv2.line(hud, (cx, cy + 10), (cx, cy + 28), reticle_col, 1)

        # Artificial Horizon Bar
        cv2.line(hud, (cx - 120, cy), (cx - 40, cy), (255, 255, 255), 1)
        cv2.line(hud, (cx + 40, cy), (cx + 120, cy), (255, 255, 255), 1)
        cv2.line(hud, (cx - 120, cy), (cx - 120, cy + 8), (255, 255, 255), 1)
        cv2.line(hud, (cx + 120, cy), (cx + 120, cy + 8), (255, 255, 255), 1)

        # Outer HUD Frame Brackets
        margin = 15
        blen = 30
        hud_col = (0, 240, 255)
        # Top-left corner
        cv2.line(hud, (margin, margin), (margin + blen, margin), hud_col, 2)
        cv2.line(hud, (margin, margin), (margin, margin + blen), hud_col, 2)
        # Top-right corner
        cv2.line(hud, (w - margin, margin), (w - margin - blen, margin), hud_col, 2)
        cv2.line(hud, (w - margin, margin), (w - margin, margin + blen), hud_col, 2)
        # Bottom-left corner
        cv2.line(hud, (margin, h - margin), (margin + blen, h - margin), hud_col, 2)
        cv2.line(hud, (margin, h - margin), (margin, h - margin - blen), hud_col, 2)
        # Bottom-right corner
        cv2.line(hud, (w - margin, h - margin), (w - margin - blen, h - margin), hud_col, 2)
        cv2.line(hud, (w - margin, h - margin), (w - margin, h - margin - blen), hud_col, 2)

        # Status text overlay
        cv2.putText(hud, "SYS: OPTICAL NAV ACTIVE", (margin + 10, margin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_col, 1, cv2.LINE_AA)
        cv2.putText(hud, "CAM-01 [PRIMARY]", (w - 150, margin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_col, 1, cv2.LINE_AA)

        return cv2.addWeighted(frame, 0.25, hud, 0.75, 0)
