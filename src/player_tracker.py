import cv2
import json
import numpy as np
from ultralytics import YOLO
from src.features import FeatureEncoder
from src.similarity_matcher import SimilarityMatcher
from src.utils import draw_tracking_info, progress_bar

class PlayerReidentificationSystem:
    def __init__(self, model_path, confidence_threshold=0.5, similarity_threshold=0.7):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = FeatureEncoder()
        self.similarity_matcher = SimilarityMatcher(similarity_threshold)
        self.player_database = {}
        self.next_player_id = 1

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        tracking_results = []
        
        for frame_number in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame)
            frame_detections = []
            
            for result in results:
                if result.boxes is not None:
                    for i in range(len(result.boxes)):
                        bbox = result.boxes.xyxy[i].cpu().numpy()
                        confidence = result.boxes.conf[i].cpu().numpy()
                        
                        if confidence >= self.confidence_threshold:
                            player_id = self._assign_player_id(frame, bbox, confidence, frame_number)
                            if player_id:
                                frame_detections.append({
                                    'player_id': player_id,
                                    'bbox': bbox.tolist(),
                                    'confidence': float(confidence),
                                    'frame': frame_number
                                })
                                if output_path:
                                    frame = draw_tracking_info(frame, bbox, player_id, confidence)
            
            tracking_results.append({'frame': frame_number, 'detections': frame_detections})
            
            if output_path:
                out.write(frame)
            
            progress_bar(frame_number + 1, total_frames)
        
        cap.release()
        if output_path:
            out.release()
        
        return {
            'tracking_results': tracking_results,
            'player_database': self.player_database,
            'total_frames': frame_number + 1
        }

    def _assign_player_id(self, frame, bbox, confidence, frame_number):
        features = self.feature_extractor.extract_features(frame, bbox)
        best_match_id = self.similarity_matcher.find_best_match(features, self.player_database)
        
        if best_match_id:
            self.player_database[best_match_id]['features'] = features
            self.player_database[best_match_id]['last_seen'] = frame_number
            return best_match_id
        else:
            new_id = self.next_player_id
            self.player_database[new_id] = {
                'features': features,
                'first_seen': frame_number,
                'last_seen': frame_number
            }
            self.next_player_id += 1
            return new_id

    def generate_report(self, results):
        report = ["=== Player Re-identification Report ==="]
        report.append(f"Total frames: {results['total_frames']}")
        report.append(f"Unique players: {len(results['player_database'])}")
        
        for player_id, info in results['player_database'].items():
            duration = info['last_seen'] - info['first_seen']
            report.append(f"\nPlayer {player_id}:")
            report.append(f"  Frames: {info['first_seen']}-{info['last_seen']} (duration: {duration})")
            report.append(f"  Features: {info['features'].shape}")
        
        return "\n".join(report)

    def save_results(self, results, output_path):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)