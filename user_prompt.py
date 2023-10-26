import video_capture as vc
import cv2
import os
import glob

# Load the Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Get the root filepath and folder datasets
root = os.getcwd()

# Ask the user for their name and see if the folder exists
prompt = input("Write your name: ")
image_folder = f'datasets/image/{prompt}'
faces_folder = f'datasets/face/{prompt}'

# Check if the folder already exists
for folder in [image_folder, faces_folder]:
    folder_path = os.path.join(root, folder)

    if not os.path.exists(folder_path):
        # If the folder doesn't exist, create it
        os.makedirs(folder_path)
        print(f"Folder created successfully at {folder_path}")
        current_id = 0
    else:
        print(f"Folder already exists at {folder_path}")

        # Use glob to get a list of files in the directory
        img_names = glob.glob(os.path.join(folder_path, '*'))
        last_filename = os.path.basename(img_names[-1])
        current_id = int(last_filename.split(".")[0]) + 1

image_path = os.path.join(root, image_folder)
face_path = os.path.join(root, faces_folder)

# Load the video capture
vc.video_cap(cascade=face_cascade, image_output=image_path, 
             face_output=face_path, total=current_id)