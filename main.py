import cv2
import os
import time
import argparse
import gdown as gd
import torch
from network import NeuralNetwork
from device import DEVICE
from transform import image_transform
from PIL import Image


def label_translator(pred_id):
    labels_dict = {
    0: 'adham', 1: 'dennis', 2: 'justin',
    3: 'kenneth', 4: 'kenneth_brille', 5: 'liza',
    6: 'liza_brille', 7: 'miriam', 8: 'steffen',
    9: 'vincent'
    }
    for key,value in labels_dict.items():
        if pred_id == key : 
            return value

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
num_labels = 10
fr_net = NeuralNetwork(num_labels).to(DEVICE)
confidence_threshold = 0.9

def face_recognition(fr_model):
    # This function takes the face recognition model and runs the video footage
    # It does the following in order:
    # 1. Loads the model  from a file (if it exists) or finds the file online via a link. The model is saved as net_state_dict within a .pt checkpoint file.
    # 2. Opens webcam and detects face using a cascade .xml file
    # 3. If faces are found, it feeds them into the face recognition model to try predict a face.
    # 4. Store the prediction as a variable. I will take it from here.
    # Open the webcam
    cap = cv2.VideoCapture(0)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame from webcam.")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection using the cascade classifier
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) containing the face
            face_roi = frame[y:y + h, x:x + w]

            # Convert the ROI np array to PIL image, resize to target  size, and convert to Tensor
            face_tensor = image_transform(face_roi).unsqueeze(0).to(DEVICE)

            # Perform face recognition using the loaded model
            with torch.no_grad():
                prediction = fr_model(face_tensor)
                # Convert to probability or "confidence value"
                confidence = torch.softmax(prediction,dim=1)
                # Get the index of label predicted
                value,preds = torch.max(confidence,1) 
                predicted_face = label_translator(preds)
            # You can take it from here and process the prediction variable as needed

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if value > confidence_threshold :
                text = "Detected Face: {}".format(predicted_face)  # Replace with your actual face label or information
            else:
                text = "Unknown Face"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

PATH_TO_URL = {
  "fr_model_test.pt":  "https://drive.google.com/file/d/15qumR5MBFtoZpq-DO-si3nA91nLbVhS_/view"
}
 
def load_model(net,model_name):
    if os.path.exists(model_name):
        chkpt = torch.load(model_name)
        net.load_state_dict(chkpt["net_state_dict"])
        return net
    else:
        if model_name in PATH_TO_URL:
            print("Downloading checkpoint...")
            gd.download(PATH_TO_URL[model_name], model_name, fuzzy=True)
            chkpt = torch.load(model_name)
            net.load_state_dict(chkpt["net_state_dict"])
            return net

        else:
            print("Error: Model doesn't exist anywhere.\nPlease train new model with train.py.")

# if __name__ == "__main__":
# Argument parser for model and cascade paths
# parser = argparse.ArgumentParser(description="Face Recognition Script")
# parser.add_argument("--model", required=True, help="Path to the face recognition model file")
# args = parser.parse_args()
            
# Load the face recognition model
model = load_model(fr_net,"fr_model_test.pt")
model.eval()
# model = load_model(fr_net,"fr_model_test.pt")
            
# Load the cascade classifier
# Run face recognition with the specified model and cascade
face_recognition(model)
        