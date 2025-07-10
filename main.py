import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.player_tracker import PlayerReidentificationSystem

def main():
    model_path = "models/yolov11_model.pt"
    input_video = "data/input/15sec_input_720p.mp4"
    output_video = "data/output/tracked_output.mp4"
    output_json = "data/output/results.json"
    
    system = PlayerReidentificationSystem(
        model_path=model_path,
        confidence_threshold=0.5,
        similarity_threshold=0.7
    )
    
    results = system.process_video(input_video, output_video)
    print(system.generate_report(results))
    system.save_results(results, output_json)

if __name__ == "__main__":
    main()