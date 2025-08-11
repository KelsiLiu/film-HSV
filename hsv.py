import cv2
import numpy as np
import sys
from tqdm import tqdm

def calculate_average_hsv(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    avg_hue = 0
    avg_saturation = 0
    avg_value = 0

    with tqdm(total=total_frames, desc="Calculating HSV", unit="frames") as pbar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_hue += np.mean(hsv_frame[:,:,0])
            avg_saturation += np.mean(hsv_frame[:,:,1])
            avg_value += np.mean(hsv_frame[:,:,2])
            pbar.update(1)

    cap.release()

    avg_hue /= total_frames
    avg_saturation /= total_frames
    avg_value /= total_frames

    return avg_hue, avg_saturation, avg_value

def calculate_scene_transition(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    prev_frame = None
    transitions = 0

    with tqdm(total=total_frames, desc="Calculating Transitions", unit="frames") as pbar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                diff = np.sum(np.abs(frame - prev_frame))
                if diff > 100000:  # Adjust threshold as needed
                    transitions += 1

            prev_frame = frame.copy()
            pbar.update(1)

    cap.release()

    return transitions / total_frames

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    movie_name = video_path.split('.')[0]  # Extracting movie name from the file name
    avg_hue, avg_saturation, avg_value = calculate_average_hsv(video_path)
    scene_transition_ratio = calculate_scene_transition(video_path)
    print("Movie:", movie_name)
    print("Average Hue:", avg_hue)
    print("Average Saturation:", avg_saturation)
    print("Average Value:", avg_value)
    print("Scene Transition Ratio:", scene_transition_ratio)

