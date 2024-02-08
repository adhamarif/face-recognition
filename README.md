# face-recognition
A project that implements deep learning methods for face detection and recognition. Two networks are used for this project, **Neural Network** and **Autoencoder**. Both models served different purposes with similar goals, which is to distinguish between known and unknown faces. We first collected the datasets of facial images of people (with consent) and used them to train our two networks. At the end, we implement our trained models back into the webcam footage which will then carry out our facial detection and recognition procedure.

## Tools used

| Library       | Version | Purpose                          |
|---------------|---------|----------------------------------|
| opencv-python | 4.8.1.78| camera & face tracking           |
| pytorch       | 2.1.0   | deep learning                    |
| matplotlib    | 3.8.2   | image plotting                   |
| pandas        | 2.1.3   | metadata management              |
| numpy         | 1.24.1  | mathematical operations on array |
| scikit-learn  | 0.24.2  | machine learning                 |
| torchvision   | 0.15.2  | image processing utilities       |
| gdown         | 5.1.0   | downloading model                |

Other dependencies are listed as well in [requirements.txt](requirements.txt)

## Installation

1 Clone this repository  
```
git clone https://github.com/adhamarif/face-recognition   
```

2. Make sure you are in the correct directory.
```
cd face-recognition  
```

3. Install the dependencies:   
```
pip install -r requirements.txt  
```

4. If you would like to train the network using graphical card, we encourage you to download the cuda version of [Pytorch](https://pytorch.org/get-started/locally/) library from its website.

## Network architecture
2 different model architectures are used to execute 2 different tasks for face recognition.

### Autoencoder
Dataset of known faces are trained by using **Autoencoder** with the aim of reconstructing the image. The logic is that at the end, if an unknown face is given as the input, the autoencoder should not be able to reconstruct the image and would return a huge loss value.

The loss value (of validation set) will then be taken as threshold value in order to distinguish known and unknown faces.

[insert AE model architecture]

### Neural Network
The same dataset of known faces are trained by a **Convolutional Neural Network** to learn the features of the images. The logic is that at the end, the model is able to recognize the features of every person in the dataset and return the right label accordingly.

The accuracy of each prediction is determined by a predefined confidence threshold that the predictions need to meet in order to be classed as a correct prediction. 

[insert Neural Network architecture]

## Usage

### Use the face recognition system with a pre-trained model.
Simply execute one of these two codes below to execute their respective face recognition tasks.

Autoencoder:
```
python main.py --model autoencoder
```
Neural Network:
```
python main.py --model network
```

### Train from scratch
If you wish to train the model from scratch with your own dataset, you need to first generate your own dataset.

#### Dataset preparation
1. Capture the images of your target faces. The images will automatically be saved under a "dataset" subdirectory of your current working directory. Run the code below :
```
python collect_data.py
```

2. Check the datasets in the face folder. If the collected images meet your standards, proceed to step 4.
3. If the collected (face) images have a lot of noises or undesired object, you might need to either:
- recollect the images dataset, **OR**
- run the cleaning dataset script below to retrieve all the (face) images detected by the *haarcascades* algorithm. After that, you may need to remove the undesired (high noise) images from your `NEW_FOLDER` manually.
```
python clean_dataset.py --original_path your_original_face_folder --cleaned_path your_new_face_folder
```
> [!IMPORTANT]
> Remember to replace your_original_face_folder and your_new_face_folder by the paths of your own folders.
4. Generate the **label.csv** file the data pipeline for the model training.
```
python label_generator.py --cleaned_path your_cleaned_dataset_path
```
> [!IMPORTANT]
> Remember to replace your_cleaned_dataset_path by the paths of your own folder.

5. Perform label encoding to your  **label.csv** file. This will generate a new file called **face_label_encoded.csv**. Execute the cells in label_encoding.ipynb (Only required for CNN model training).

> [!IMPORTANT]
> Remember to check if you have the correct one-hot-encoded values in your DataFrame columns before saving the **face_label_encoded.csv**.

#### Train the model
Autoencoder:
```
python train_autoencoder.py --images_folder your_faces_dataset_folder
```

Neural Network:

```
python train_network.py --images_folder your_faces_dataset_folder
```
> [!IMPORTANT]
> Remember to replace your_faces_dataset_folder by the paths to your own folders. They are the same path for both models.

#### Run the facial recognition system with the new model
1. Both the trained autoencoder and the trained CNN model will be saved under a  'models' directory.
2. You can specify which one to use using `--model` argument when running the facial recognition system.

Autoencoder:
```
python main.py --model autoencoder
```
Neural Network:
```
python main.py --model network
```

## Results
### Autoencoder
Input image:

Output image:

### Face recognition system
