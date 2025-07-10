# Player-Re-identification

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red)](https://ultralytics.com/yolov8)
[![Torchreid](https://img.shields.io/badge/Torchreid-v1.4.0-green)](https://github.com/KaiyangZhou/deep-person-reid)

Real-time player tracking system that maintains consistent identities across broadcast video frames using visual re-identification.


## Features ✨

- **YOLOv8 Detection**: Custom-trained model for player/ball detection
- **Visual Re-ID**: `osnet_x1_0` model for appearance embeddings
- **Two-Stage Tracking**:
  - Short-term: ByteTrack for frame-to-frame association
  - Long-term: Cosine similarity matching (threshold=0.7)
- **Gallery Management**:
  - `active_tracks`: Currently visible players
  - `inactive_gallery`: Players who exited frame (50-frame memory)

## Installation 🛠️

```bash
git clone https://github.com/VaishnaviSrivastava033/Player-Re-identification.git
cd Player-Re-identification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running The Code 
# 1. Testing
```bash
python test_imports.py
```

# 2.Run
```bash
python main.py
```

## Usage 🚀

### Detection Only:
```bash
python detect.py --source input/video.mp4 --weights models/best.pt
```

### Full Tracking + Re-ID
python track_reid.py \
    --source input/video.mp4 \
    --detection-weights models/best.pt \
    --reid-weights models/osnet_x1_0.pth

### Project Structure 📂
player-reidentification\
├── src\
│   ├── __init__.py
│   ├── player_tracker.py
│   ├── feature_extractor.py
│   ├── similarity_matcher.py
│   └── utils.py
├── models\
│   └── yolov11_model.pt
├── data\
│   ├── input\
│   │   └── 15sec_input_720p.mp4
│   └── output\
├── venv\
├── requirements.txt
├── config.py
└── main.py

### Results 📊
Metric	Performance
Short-term ID accuracy	92% (@50 frames)
Re-ID after occlusion	68% success
Processing speed	18 FPS (RTX 3060)
Output samples:

output/detected_video_custom.mp4 - Raw detections

output/tracked_video_reid_final.mp4 - With Re-ID

### Connect 🤝
[Github](https://img.shields.io/badge/GitHub-VaishnaviSrivastava033-blue)
[Linkeldn](https://img.shields.io/badge/https://www.linkedin.com/in/vaishnavi-srivastava033/-blue)





