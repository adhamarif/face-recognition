import cv2
import os
import time
import argparse
import gdown as gd
import torch
from network.network import NeuralNetwork
from network.device import DEVICE
from network.transform import image_transform as net_img_transform
from PIL import Image
from torchvision.transforms import transforms
import torch.nn as nn
from autoencoder.autoencoder_net import Network

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
num_labels = 12
fr_net = NeuralNetwork(num_labels).to(DEVICE)
confidence_threshold = 0.9

ae_net = Network().to(DEVICE)

PATH_TO_URL = {
  "fr_model_best.pt":  "https://drive.google.com/file/d/1YIgIkFDk_P_j9iSnnxSt0F-bqMavxJDb/view",
  "autoencoder_best_model.pth": "https://drive.google.com/file/d/1-P3wPTDgb2Xhw9NCnrfpUHNxXZl00_Jy/view?usp=sharing" #AE
}

# Define image transformation
image_transform = transforms.Compose([
    transforms.Resize((320, 320)),  # Resize the image
    transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
])

def label_translator(pred_id):
    labels_dict = {
    0: 'adham', 1: 'dennis', 2: 'justin',
    3: 'kenneth', 4: 'kenneth_brille', 5: 'liza',
    6: 'liza_brille', 7: 'miriam', 8: 'steffen',
    9: 'syahid', 10: 'syahid_brille', 11: 'vincent'
    }
    for key,value in labels_dict.items():
        if pred_id == key : 
            return value


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
            face_tensor = net_img_transform(face_roi).unsqueeze(0).to(DEVICE)

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
        

def ae_face_recognition(ae_model, loss_threshold=0.01):
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
    mse_loss = nn.MSELoss()
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

             # Convert the NumPy array to a PIL Image
            face_pil_image = Image.fromarray(face_roi)

            # Apply the transformations
            transformed_image = image_transform(face_pil_image)

            # Convert the transformed image to a PyTorch tensor
            face_tensor = (transformed_image).unsqueeze(0).to(DEVICE)

            # Perform face recognition using the loaded model
            with torch.no_grad():
                reconstructed = ae_model(face_tensor)
            
            loss = mse_loss(face_tensor, reconstructed)
            # You can take it from here and process the prediction variable as needed

            if loss < loss_threshold :
                text = f"Known Face {loss:.4f}"
                color = (0, 255, 0)  # Green color
            else:
                text = f"Unknown Face {loss:.4f}"
                color = (0, 0, 255)  # Red color
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
 
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
model = load_model(fr_net,"fr_model_best.pt")
model.eval()

# Load the autoencoder face recognition model
model = load_model(ae_net,"autoencoder_best_model.pth")
model.eval()
            
# Load the cascade classifier
# Run face recognition with the specified model and cascade
ae_face_recognition(model)
        