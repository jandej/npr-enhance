
# Number Plate Recognition (NPR)

A Python-based Number Plate Recognition system that processes video files to enhance, detect, track, and recognize license plates from basd quality video.

## Features

*   **License Plate Detection:** Utilizes OpenCV and a Haar Cascade classifier to detect license plates in video frames.
*   **Plate Tracking:** Implements a custom tracker to follow individual plates across multiple frames, associating detections of the same plate over time.
*   **Image Enhancement:** Improves the quality of the license plate image for OCR by aligning and averaging multiple frames of a tracked plate. This reduces noise and increases recognition accuracy.
*   **OCR:** Extracts text from the enhanced plate image using Tesseract OCR.
*   **Test Video Generation:** Includes a script to create synthetic test videos with moving license plates for easy testing and demonstration.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Tesseract OCR:**
    This project uses Tesseract for OCR. You need to install it on your system.

    *   **On Debian/Ubuntu:**
        ```bash
        sudo apt-get install tesseract-ocr
        ```
    *   **On macOS (using Homebrew):**
        ```bash
        brew install tesseract
        ```
    *   **On Windows:**
        Download and run the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page. Make sure to add the Tesseract installation directory to your system's `PATH`.

4.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Processing a Video

To process a video and recognize license plates, run the `process_video.py` script:

```bash
python process_video.py --video /path/to/your/video.mp4
```

To see the video output with bounding boxes and tracking information, use the `--debug` flag:

```bash
python process_video.py --video /path/to/your/video.mp4 --debug
```

### Generating a Test Video

To generate a test video file named `test_video.mp4`, run the `create_test_video.py` script:

```bash
python create_test_video.py
```

This will create a video with a synthetic license plate that can be used to test the processing script.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
