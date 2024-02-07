import cv2
import os
import glob

# Load the Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

ORIGINAL_IMAGE_PATH = r'D:\daisy_dataset\image\steffen'
NEW_FOLDER = r"D:\clean_dataset\steffen"
total = 0

# Check if the TEST_FOLDER exists, if not, create one
if not os.path.exists(NEW_FOLDER):
    os.makedirs(NEW_FOLDER)
    print(f"Folder '{NEW_FOLDER}' created successfully.")

# Use glob to get a list of files in the directory
img_names = glob.glob(os.path.join(ORIGINAL_IMAGE_PATH, '*'))

#print(img_names)
total_noise = 0
for idx in img_names:
    img = cv2.imread(idx, cv2.IMREAD_COLOR)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    if len(faces) > 1:
        total_noise += len(faces)
        print(f"Total images need to be reviewed: {total_noise}")

    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        # Cropping original frame according to region of face detection, removing the bounding box border
        face_img = img[y+2:y+h-2,x+2:x+w-2]


        # Writes the cropped face into the desired directory
        face_path = os.path.sep.join([NEW_FOLDER, f"{str(total).zfill(5)}.png"])
        cv2.imwrite(face_path, face_img)
            
        total += 1