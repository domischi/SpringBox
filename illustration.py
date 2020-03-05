# Adapted from answer by BoboDarph (Stackoverflow: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python)

import cv2
import os

def generate_video_from_png(image_folder):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    if len(images)>0:
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video_path=f'{image_folder}/video.avi'

        video = cv2.VideoWriter(video_path, 0, 30, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        return video_path
