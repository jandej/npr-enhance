
import cv2
import numpy as np
import random

# --- Configuration ---
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_DURATION_FRAMES = 150
VIDEO_FILENAME = "test_video.mp4"
PLATE_TEXT = "GEM-IN1"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Plate Image Generation ---
# Create a blank image for the number plate
plate_img = np.zeros((60, 200, 3), dtype=np.uint8)
plate_img.fill(255) # White background

# Add text to the plate
font_scale = 1.5
font_thickness = 3
text_size, _ = cv2.getTextSize(PLATE_TEXT, FONT, font_scale, font_thickness)
text_x = (plate_img.shape[1] - text_size[0]) // 2
text_y = (plate_img.shape[0] + text_size[1]) // 2
cv2.putText(plate_img, PLATE_TEXT, (text_x, text_y), FONT, font_scale, (0, 0, 0), font_thickness)

# Add a black border
cv2.rectangle(plate_img, (0, 0), (plate_img.shape[1]-1, plate_img.shape[0]-1), (0, 0, 0), 3)


# --- Video Generation ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 30.0, (VIDEO_WIDTH, VIDEO_HEIGHT))

# Initial position of the plate
x_pos = 100
y_pos = 200

print(f"Generating video: {VIDEO_FILENAME}...")

for frame_num in range(VIDEO_DURATION_FRAMES):
    # Create a black background frame
    frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    frame.fill(50) # Dark grey background

    # --- Simulate Movement ---
    # Move the plate slightly and randomly
    x_pos += random.randint(-1, 1)
    y_pos += random.randint(-1, 1)

    # Keep plate within bounds
    if x_pos < 0: x_pos = 0
    if y_pos < 0: y_pos = 0
    if x_pos + plate_img.shape[1] > VIDEO_WIDTH: x_pos = VIDEO_WIDTH - plate_img.shape[1]
    if y_pos + plate_img.shape[0] > VIDEO_HEIGHT: y_pos = VIDEO_HEIGHT - plate_img.shape[0]

    # Get the region of interest (ROI) where the plate will be placed
    roi = frame[y_pos:y_pos + plate_img.shape[0], x_pos:x_pos + plate_img.shape[1]]

    # --- Add Noise to the Plate ---
    noisy_plate = plate_img.copy()
    # Add some salt-and-pepper noise
    noise = np.random.randint(0, 50, noisy_plate.shape, dtype=np.int16)
    noisy_plate = np.clip(noisy_plate.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Place the noisy plate onto the frame
    frame[y_pos:y_pos + plate_img.shape[0], x_pos:x_pos + plate_img.shape[1]] = noisy_plate


    # Write the frame to the video
    video_writer.write(frame)

video_writer.release()
print("Video generation complete.")
