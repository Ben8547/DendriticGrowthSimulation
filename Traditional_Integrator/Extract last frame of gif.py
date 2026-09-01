import os
import cv2
from PIL import Image

def extract_last_frame_mp4(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"No frames found in {video_path}")
        cap.release()
        return

    # Set position to last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    ret, frame = cap.read()
    if ret:
        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.save(output_path)
        print(f"Saved last frame of {video_path} -> {output_path}")
    else:
        print(f"Failed to read last frame of {video_path}")

    cap.release()


if __name__ == "__main__":
    current_dir = os.getcwd()
    base_name = '/seed-43095-dendrite_growth_simulation-pulses-newcurrent-20_electrons-visc-1.0_vdW-1.0_Ly-20.0-1000particles.mp4'
    output_path = os.path.join(current_dir, f"{base_name}_last_frame.png")
    video_path = current_dir+base_name
    extract_last_frame_mp4(video_path, output_path)
    
    if False:
        for filename in os.listdir(current_dir):
            if filename.lower().endswith(".mp4"):
                video_path = os.path.join(current_dir, filename)

                # Create output filename
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(current_dir, f"{base_name}_last_frame.png")

                extract_last_frame_mp4(video_path, output_path)
