import cv2
import sys

def draw_tracking_info(frame, bbox, player_id, confidence):
    """Draw bounding box and player ID on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    label = f'Player {player_id} ({confidence:.2f})'
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                 (x1 + label_size[0], y1), (0, 255, 0), -1)
    
    cv2.putText(frame, label, (x1, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def progress_bar(current, total, bar_length=50):
    """Display progress bar"""
    progress = current / total
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    sys.stdout.write(f'\rProgress: |{bar}| {progress:.1%} ({current}/{total})')
    sys.stdout.flush()
    
    if current == total:
        print()