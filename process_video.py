
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import argparse
from math import sqrt

# --- Configuration ---
# Path to the Haar Cascade model
CASCADE_PATH = "/home/jandej/Development/NPR/haarcascade_russian_plate_number.xml"

# --- Tracker Class ---
class PlateTracker:
    def __init__(self, tracker_id, initial_bbox):
        self.id = tracker_id
        self.bbox = initial_bbox
        self.frames = []
        self.frames_since_seen = 0
        self.recognized_text = None

    def get_center(self):
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

# --- Main Processing Function ---
def process_video(video_path, debug=False, use_gpu=False):
    # Check for CUDA availability if GPU is requested
    if use_gpu:
        try:
            cv2.cuda.getCudaEnabledDeviceCount()
            print("GPU is available. Using CUDA for processing.")
        except cv2.error:
            print("Error: OpenCV is not compiled with CUDA support. Falling back to CPU.")
            use_gpu = False

    # Load the Haar Cascade for plate detection
    if use_gpu:
        try:
            plate_cascade = cv2.cuda.CascadeClassifier_create(CASCADE_PATH)
        except AttributeError:
            print("Error: Your OpenCV installation does not support GPU-accelerated cascade classifiers.")
            print("Falling back to CPU.")
            use_gpu = False
            plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    else:
        plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    if plate_cascade.empty():
        print(f"Error: Could not load Haar Cascade from {CASCADE_PATH}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    trackers = []
    next_tracker_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect plates
        if use_gpu:
            gpu_gray_frame = cv2.cuda_GpuMat()
            gpu_gray_frame.upload(gray_frame)
            detected_plates_gpu = plate_cascade.detectMultiScale(gpu_gray_frame)
            detected_plates = detected_plates_gpu.download()
            if detected_plates is None:
                detected_plates = []
            else:
                detected_plates = detected_plates[0]
        else:
            detected_plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # --- Tracking Logic ---
        matched_tracker_ids = []

        for (x, y, w, h) in detected_plates:
            current_plate_center = (x + w // 2, y + h // 2)
            best_match_id = -1
            min_dist = float('inf')

            # Find the closest existing tracker
            for tracker in trackers:
                if tracker.id in matched_tracker_ids:
                    continue
                dist = sqrt((tracker.get_center()[0] - current_plate_center[0])**2 + \
                            (tracker.get_center()[1] - current_plate_center[1])**2)
                
                if dist < 50 and dist < min_dist: # 50 pixels is our matching threshold
                    min_dist = dist
                    best_match_id = tracker.id

            # If a match is found, update the tracker
            if best_match_id != -1:
                tracker = next(t for t in trackers if t.id == best_match_id)
                tracker.bbox = (x, y, w, h)
                tracker.frames_since_seen = 0
                matched_tracker_ids.append(best_match_id)
                # Add the cropped plate if not yet recognized
                if not tracker.recognized_text:
                    tracker.frames.append(gray_frame[y:y+h, x:x+w])
            # Otherwise, create a new tracker
            else:
                new_tracker = PlateTracker(next_tracker_id, (x, y, w, h))
                next_tracker_id += 1
                new_tracker.frames.append(gray_frame[y:y+h, x:x+w])
                trackers.append(new_tracker)
                matched_tracker_ids.append(new_tracker.id)

        # --- Process and Clean Up Trackers ---
        for tracker in trackers:
            # If a tracker wasn't seen, increment its counter
            if tracker.id not in matched_tracker_ids:
                tracker.frames_since_seen += 1

            # If a tracker has enough frames, try to recognize it
            if len(tracker.frames) >= 10 and not tracker.recognized_text:
                print(f"Tracker {tracker.id}: Collected 10 frames. Attempting enhancement and OCR.")
                
                # --- Image Enhancement ---
                # 1. Resize all images to match the first one
                reference_shape = tracker.frames[0].shape
                resized_frames = [cv2.resize(f, (reference_shape[1], reference_shape[0])) for f in tracker.frames]
                
                # 2. Align images to the first frame
                aligned_frames = [resized_frames[0]]
                warp_mode = cv2.MOTION_TRANSLATION
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-10)

                for i in range(1, len(resized_frames)):
                    try:
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        (cc, warp_matrix) = cv2.findTransformECC(resized_frames[0], resized_frames[i], warp_matrix, warp_mode, criteria)
                        aligned_frame = cv2.warpAffine(resized_frames[i], warp_matrix, (reference_shape[1], reference_shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        aligned_frames.append(aligned_frame)
                    except cv2.error:
                        # If alignment fails, just skip this frame
                        continue

                # 3. Average the aligned frames
                if len(aligned_frames) > 1:
                    enhanced_plate = np.mean(aligned_frames, axis=0).astype(np.uint8)
                else:
                    enhanced_plate = aligned_frames[0]

                # 4. Final image prep for OCR
                _, enhanced_plate = cv2.threshold(enhanced_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                enhanced_plate = cv2.bitwise_not(enhanced_plate)

                # --- OCR ---
                # Use Tesseract to get the text and confidence
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                data = pytesseract.image_to_data(enhanced_plate, config=custom_config, output_type=Output.DICT)
                
                text = "".join(data['text']).strip()
                confidences = [int(c) for c in data['conf'] if int(c) != -1]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                if text and len(text) > 3:
                    tracker.recognized_text = text
                    print(f"===================================")
                    print(f"Plate Recognized! Text: {text}, Confidence: {avg_confidence:.2f}%")
                    print(f"===================================")
                    if debug:
                        cv2.imshow(f"Enhanced Plate ID: {tracker.id}", enhanced_plate)
                else:
                    # If recognition fails, clear frames to try again
                    tracker.frames = []

            # Draw bounding boxes on the frame for debugging
            if debug:
                x, y, w, h = tracker.bbox
                color = (0, 255, 0) if tracker.recognized_text else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"ID: {tracker.id}"
                if tracker.recognized_text:
                    label = f"ID: {tracker.id} - {tracker.recognized_text}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Remove old trackers
        trackers = [t for t in trackers if t.frames_since_seen < 15]

        if debug:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if debug:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video to recognize number plates.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode to show video output.")
    parser.add_argument("--gpu", action='store_true', help="Enable GPU acceleration for processing.")
    args = parser.parse_args()

    process_video(args.video, args.debug, args.gpu)

