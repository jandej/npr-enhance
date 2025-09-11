# npr-enhance

The system is composed of the following main components:
   - Video Processing: Reads a video file frame by frame using OpenCV.
   - Plate Detection: Utilizes a Haar Cascade classifier (haarcascade_russian_plate_number.xml) to detect potential license plate regions in each frame.
   - Plate Tracking: A custom PlateTracker class is implemented to track detected plates across multiple frames. This allows for associating detections of the same plate over time.
   - Image Enhancement: To improve the accuracy of Optical Character Recognition (OCR), the system collects multiple frames of a tracked plate. These frames are then aligned and averaged to create a higher quality, less noisy image of the plate.
   - OCR: The enhanced plate image is passed to Tesseract for character recognition.
   - Test Video Generation: A script (create_test_video.py) is included to generate synthetic test videos containing a moving license plate. This is useful for testing the NPR pipeline without requiring real-world footage.

The main script (process_video.py) takes a video file as input and outputs the recognized license plate text to the console. A debug mode is available to visualize the detection and tracking process.
