import cv2
import numpy as np

class VisualOdometry:
    """Monocular visual odometry using essential matrix + recoverPose.
    Tracks relative 2D position (x, y), yaw heading, trajectory history, and distance traveled.
    """
    def __init__(self):
        self.prev_kp = None
        self.prev_des = None
        self.orb = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.total_distance = 0.0
        self.frames = 0
        self.history = [(0.0, 0.0, 0.0)]
        self.last_kp = None
        self.last_good_matches = []

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        self.last_kp = kp
        
        matches = 0
        good = 0
        inliers = 0
        conf = 0.0
        dx = dy = dyaw = 0.0
        
        h, w = gray.shape
        fx = 0.9 * w
        fy = fx
        cx = w / 2
        cy = h / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        
        if self.prev_des is not None and des is not None:
            pairs = self.bf.knnMatch(self.prev_des, des, k=2)
            goodm = [m for m, n in pairs if len(pairs) > 0 and m.distance < 0.72 * n.distance]
            self.last_good_matches = goodm
            matches = len(pairs)
            good = len(goodm)
            
            if good >= 12:
                p0 = np.float32([self.prev_kp[m.queryIdx].pt for m in goodm])
                p1 = np.float32([kp[m.trainIdx].pt for m in goodm])
                E, mask = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.2)
                
                if E is not None:
                    _, R, t, mask2 = cv2.recoverPose(E, p0, p1, K, mask=mask)
                    inliers = int(mask2.sum() / 255) if mask2 is not None else 0
                    conf = float(np.clip(inliers / max(good, 1), 0.0, 1.0))
                    
                    # Bounded relative step
                    step = float(np.clip(np.linalg.norm(t), 0.01, 0.20))
                    dx = float(t[0, 0]) * step
                    dy = float(t[2, 0]) * step
                    dyaw = float(np.arctan2(R[1, 0], R[0, 0]))
                    
                    self.x += dx
                    self.y += dy
                    self.yaw += dyaw
                    
                    dist_step = np.sqrt(dx**2 + dy**2)
                    self.total_distance += dist_step
                    self.history.append((self.x, self.y, self.yaw))
                    
                    # Keep history capped at 1000 points
                    if len(self.history) > 1000:
                        self.history.pop(0)

        self.prev_kp, self.prev_des = kp, des
        self.frames += 1
        
        return dict(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            matches=good,
            inliers=inliers,
            confidence=conf,
            total_distance=self.total_distance,
            history=self.history
        )

    def draw_features(self, frame):
        """Overlay feature keypoints for technical visual inspection."""
        out = frame.copy()
        if self.last_kp is not None:
            for kp in self.last_kp[::3]:  # sub-sample keypoints
                x, y = map(int, kp.pt)
                cv2.circle(out, (x, y), 2, (0, 255, 255), -1)
        return out

    def reset(self):
        self.prev_kp = None
        self.prev_des = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.total_distance = 0.0
        self.frames = 0
        self.history = [(0.0, 0.0, 0.0)]
        self.last_kp = None
        self.last_good_matches = []
