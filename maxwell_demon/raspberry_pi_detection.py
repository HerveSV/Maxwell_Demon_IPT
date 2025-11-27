import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys



THRESH = 145
MIN_AREA = 20



def is_there_particle(frame_gray):
     # threshold bright balls
    _, bw = cv2.threshold(frame_gray, THRESH, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # find blobs
    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue

        # True if contour above threshold area is detected
        return True

    return False





# Define crop window

# for 40mm wide window
'''x1, x2 = 40, 150
y1, y2 = 80, 190'''

# for 20mm wide window
'''x1, x2 = 67, 122
y1, y2 = 135, 190'''

# 20mm wide, 10mm high. 5mm offset from bottom
'''x1, x2 = 67, 123
y1, y2 = 148, 176'''

# 20mm wide, 10mm high. 1mm offset from bottom
'''x1, x2 = 67, 123
y1, y2 = 159, 187'''

# 20mm wide, 10mm high. 2mm offset from bottom
'''x1, x2 = 67, 123
y1, y2 = 157, 185'''

# 40mm wide, 10mm high. 2mm offset from bottom
x1, x2 = 40, 150
y1, y2 = 157, 185




# --------------- Main program loop ------------

print("Loading video")

video_path = "/Users/hervesv/Documents/CloudDrive/IPT/Maxwell_Demon_2025/videos/13-11-25/1.75g/cropped/1.75g@15Hz.mov"
print(f"Current video {video_path}")
cap = cv2.VideoCapture(video_path)

print("Video loaded")

# ---- Show first frame with crop window ----
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Error reading first frame")

# Draw the crop rectangle
cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show the frame in a window
cv2.imshow("First frame with crop window", first_frame)
cv2.waitKey(0)   # Wait until you press a key
cv2.destroyAllWindows()

# Reset the video to frame 0 so main loop starts fresh
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# do stuff

print("Analysing frames")
if not cap.isOpened():
    raise RuntimeError("Could not open video file.")

times_taken = []
particle_in_frame = []

frame_idx = 0


while True:

    time1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    crop = gray[y1:y2, x1:x2]
    particle_found = is_there_particle(crop)
    time2 = time.time()

    times_taken.append(time2-time1)
    particle_in_frame.append(particle_found)
    
    frame_idx += 1

cap.release()

times_taken = np.array(times_taken)
print(f"Average time taken {np.mean(times_taken)*1e3} ± {np.std(times_taken)*1e3} ms")
all_True = True
red_frames = 0
for i in range(len(particle_in_frame)):
    if particle_in_frame[i] == False: 
        all_True = False
        red_frames += 1
        #print(f"No particles found in frame {i}")

if all_True: print("All is good, particles detected for all frames")

print(f"Number of red frames {red_frames}")



output_path = "videos/annotated_video.mp4"
if True:
    # =====================================================
    # =============== LOOP 2: DRAW + SAVE =================
    # =====================================================

    cap = cv2.VideoCapture(video_path)

    # Output video writer setup
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{fps = }")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    red = (0, 0, 255)
    green = (0, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the crop window rectangle

        border_color = green if particle_in_frame[frame_idx] else red
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                    border_color,  # green
                    2)            # thickness

        # Write to output video
        out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()



    print("Annotated video saved as:", output_path)