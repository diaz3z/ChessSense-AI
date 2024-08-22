# Chess Piece Detection using YOLOv8

This project implements a chess piece detection system using YOLOv8 object detection model. It processes video input of chess games, identifies the pieces on the board, and outputs a video with bounding boxes and labels for each detected piece.

## Features

- Real-time chess piece detection in video streams
- Utilizes YOLOv8 for accurate object detection
- Draws bounding boxes and labels on detected chess pieces
- Outputs processed video with visual annotations

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLOv8
- Stockfish (for potential future integration with chess engines)
## Installation

1. Clone this repository:

```bash
git clone https://github.com/diaz3z/ChessSense-AI.git

cd ChessSense-AI
```
2. Install the required packages:  

```bash
pip install opencv-python ultralytics stockfish
```
3. Download the trained YOLOv8 model weights and place them in the `runs/detect/train6/weights/` directory.

## Usage

1. Place your input chess game video in the `Video/` directory.

2. Run the script:
```bash
python save.py
```

3. The processed video will be saved as `output_video.mp4` in the project directory.

4. Or to run locally Run this script:
```bash
python chessboard+chesspieces localize.py
```

## How it works

1. The script loads a pre-trained YOLOv8 model for chess piece detection.
2. It processes the input video frame by frame.
3. For each frame, it detects chess pieces and their positions.
4. Bounding boxes and labels are drawn on the detected pieces.
5. The processed frames are compiled into an output video.

## Future Improvements

- Integration with a chess engine for move analysis
- Real-time board state tracking
- Support for live video input from cameras
- Web interface for easy usage

## Contributing

Contributions to improve the project are welcome. Please feel free to fork the repository and submit pull requests.

## License

[MIT License](LICENSE)

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Stockfish](https://stockfishchess.org/)
## Screenshots

![App Screenshot] https://github.com/diaz3z/ChessSense-AI/blob/dc7ecfdbded1d469007c3bcdb062948a5abe3c65/Images/Screenshot%202024-05-27%20162428.png

