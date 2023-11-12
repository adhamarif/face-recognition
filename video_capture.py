import cv2
import os

def video_cap(cascade, image_output, face_output, total=0):
    """
    This function runs the image capturing and face cropping process 
    of generating the facial recognition dataset.
    """
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read the frame from the video stream
        ret, frame = cap.read()
        # Flip the camera to get mirror image
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the copy frame to save as original image
        copy_frame = frame.copy()

        # Detect eyes in the grayscale frame
        faces = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (x, y, w, h) in faces:
            # Cropping original frame according to region of face detection, removing the bounding box border
            face_img = frame[y+2:y+h-2,x+2:x+w-2]
            # Display bounding box in display window
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame with detected eyes
        cv2.imshow('Detected face', frame)

        if cv2.waitKey(1) & 0xFF == ord('k'):

            # Writes the original image into the desired directory
            image_path = os.path.sep.join([image_output, f"{str(total).zfill(5)}.png"])
            cv2.imwrite(image_path, copy_frame)
            
            # Writes the cropped face into the desired directory
            face_path = os.path.sep.join([face_output, f"{str(total).zfill(5)}.png"])
            cv2.imwrite(face_path, face_img)
                
            total += 1
            
            print(f"{total} images captured and saved.")

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
