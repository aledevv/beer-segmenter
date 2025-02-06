# Video Beer Segmenter

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview
Project for the course of Signal Image Video regarding a simple segmenter to detect beer foam through image processing techniques. This task was previously solved using AI.

## Notebook
The project includes a Jupyter Notebook that provides an in-depth explanation of the methodology used for video processing. It demonstrates:
- **Preprocessing**: How video frames are extracted and converted for processing.
- **Segmentation Techniques**: Methods applied to identify beer in the video frames.
- **Differences with AI solution**: discussion on the differences of the two approaches

To run the notebook:
```sh
jupyter notebook beer_segmenter.ipynb
```

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/video-beer-segmenter.git
cd video-beer-segmenter
```

### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Running the Video Beer Segmenter
Modify the `video_beer_segmenter.py` script to specify the input video path and the output video name inside the `main` function:

```python
if __name__ == "__main__":
    input_video = "path/to/input_video.mp4"  # Modify this
    output_video = "output_video.mp4"  # Modify this
    process_video(input_video, output_video)
```

Then run:
```sh
python video_beer_segmenter.py
```

## Dependencies
This project requires the following packages:
- numpy
- opencv-python
- matplotlib
- Pillow
- tqdm

## License
MIT License

## Author
Alessandro De Vidi - [aledevv](https://github.com/aledevv)



    

