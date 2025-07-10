import cv2
import numpy as np

class FeatureEncoder:
    def __init__(self):
        self.target_size = (64, 128)  # Standard person re-id size
    
    def extract_features(self, frame, bbox):
        """Extract visual features from player region"""
        x1, y1, x2, y2 = map(int, bbox)
        
        player_region = frame[y1:y2, x1:x2]
        if player_region.size == 0:
            return np.zeros(512)
        
        player_region = cv2.resize(player_region, self.target_size)
        
        color_features = self._extract_color_features(player_region)
        texture_features = self._extract_texture_features(player_region)
        
        features = np.concatenate([color_features, texture_features])
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def _extract_color_features(self, region):
        """Extract color histogram features"""
        features = []
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        return np.array(features)
    
    def _extract_texture_features(self, region):
        """Extract texture features using gradients"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        features = [
            np.std(grad_x),
            np.std(grad_y),
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y))
        ]
        
        return np.array(features)