import cv2
import math
import numpy as np

def build_costmap(free, obstacles, grid=(36, 48)):
    gh, gw = grid
    h, w = free.shape
    free_small = cv2.resize((free > 0).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    
    # Lower free-space confidence becomes high cost
    cost = (1.0 - free_small) * 6.0
    yy, xx = np.mgrid[0:gh, 0:gw]
    
    # Mild preference for forward progression
    cost += 0.18 * np.maximum(0, (yy - (gh * 0.65)) / gh)
    
    for ob in obstacles:
        x1, y1, x2, y2 = ob["bbox"]
        gx1 = int(x1 / w * gw)
        gx2 = int(x2 / w * gw)
        gy1 = int(y1 / h * gh)
        gy2 = int(y2 / h * gh)
        pad = int(1 + 3 * ob["risk"])
        xa, xb = max(0, gx1 - pad), min(gw, gx2 + pad + 1)
        ya, yb = max(0, gy1 - pad), min(gh, gy2 + pad + 1)
        cost[ya:yb, xa:xb] += 18.0 * ob["risk"]
        
    # Smooth cost field
    cost = cv2.GaussianBlur(cost.astype(np.float32), (5, 5), 0)
    return cost

def trajectory_score(cost, start, heading, steer, speed, horizon=16, target_bias=0.0):
    gh, gw = cost.shape
    r, c = start
    total = 0.0
    pts = []
    angle = heading + steer
    
    for i in range(1, horizon + 1):
        c += math.sin(angle) * 0.75 * speed
        r -= math.cos(angle) * 0.75 * speed
        ri = int(round(r))
        ci = int(round(c))
        
        if not (0 <= ri < gh and 0 <= ci < gw):
            return 1e6, []
            
        pts.append((ri, ci))
        total += float(cost[ri, ci])
        angle += steer * 0.035
        
    # Target goal bias (steer towards target direction preference)
    goal_cost = abs(steer - target_bias) * 0.8
    smooth = abs(steer) * 1.2 + abs(speed - 0.75) * 0.3 + goal_cost
    return (total / horizon) + smooth, pts

def dynamic_window_plan(cost, target_bias=0.0):
    """Dynamic Window Approach (DWA) planner.
    Evaluates candidate steering/speed arcs and returns optimal path + candidate paths.
    """
    gh, gw = cost.shape
    start = (gh - 2, gw // 2)
    candidates = []
    
    for steer in np.linspace(-0.85, 0.85, 13):
        for speed in np.linspace(0.35, 1.0, 5):
            score, path = trajectory_score(cost, start, 0, steer, speed, target_bias=target_bias)
            if path:
                candidates.append((score, steer, speed, path))
                
    if not candidates:
        return [], 0.0, 0.0, []
        
    candidates.sort(key=lambda x: x[0])
    best_score, best_steer, best_speed, best_path = candidates[0]
    all_candidate_paths = [c[3] for c in candidates[1:12]]
    
    return best_path, best_steer, best_speed, all_candidate_paths

def command(steer, speed, risk, vo_conf):
    if risk >= 0.80:
        return "EMERGENCY STOP", 0.0
    if risk >= 0.58:
        return "OBSTACLE AVOIDANCE", max(0.15, speed * 0.35)
    if steer < -0.22:
        return "BEARING LEFT", speed
    if steer > 0.22:
        return "BEARING RIGHT", speed
    return "FORWARD FULL", speed

def draw_path(frame, best_path, candidate_paths=None, draw_candidates=True):
    """Renders candidate trajectory arcs and optimal planned path onto frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    gh, gw = 36, 48
    
    # Draw candidate trajectory arcs in translucent blue/cyan
    if draw_candidates and candidate_paths:
        for cand in candidate_paths:
            cand_pts = [(int(c / gw * w), int(r / gh * h)) for r, c in cand]
            if len(cand_pts) > 1:
                cv2.polylines(out, [np.array(cand_pts, np.int32)], False, (200, 150, 50), 1, cv2.LINE_AA)

    # Draw optimal trajectory path in glowing yellow/cyan
    pts = [(int(c / gw * w), int(r / gh * h)) for r, c in best_path]
    if len(pts) > 1:
        # Outer glow
        cv2.polylines(out, [np.array(pts, np.int32)], False, (0, 255, 255), 4, cv2.LINE_AA)
        # Inner core
        cv2.polylines(out, [np.array(pts, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Target destination point marker
        target_pt = pts[-1]
        cv2.circle(out, target_pt, 6, (0, 255, 0), -1)
        cv2.circle(out, target_pt, 9, (255, 255, 255), 1)

    # UGV Base Origin Marker
    cv2.circle(out, (w // 2, h - 12), 8, (255, 0, 255), -1)
    return out
