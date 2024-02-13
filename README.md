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

1. Clone this repository  
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

The autoencoder architecture consists of the following components:
- **Down**: A layer consists of convolution, maxpooling, ReLU and batch normalization resulting in a reduction in spatial dimensions.
- **Flatten**: A layer that flattens the output from the last convolutional layer to prepare it for fully connected layers.
- **Linear**: Fully connected layers. 
- **Unflatten** : A layer that reshape the vector back to its original shape
- **Up**: A layer consists of convolution transpose, ReLU and batch normalization to reconstruct the image from lower dimensional embeddings to its original shape.

<img src="https://github.com/adhamarif/face-recognition/blob/main/readme_graphics/AE.png">

- **Encoder** : Down layers are repeated 6 times in sequence from to learn the feature from the input image (320, 320) to the lower dimensional shape (at the bottleneck). At the end of the encoder, a fully connected layer is added for the model to learn about every information in every pixels.
- **Decoder** : A fully connected layer is added for the model to learn about every information in every pixels. Then, the Up layers are repeated 6 times to reconstruct the image back to its original dimension (320, 320). The last layer of Decoder is designed that it has no batch normalization and used Sigmoid as the activation function. The reason is because the final output should be retained as raw value and should not be normalized. 

### Neural Network
The same dataset of known faces are trained by a **Convolutional Neural Network** to learn the features of the images. The logic is that at the end, the model is able to recognize the features of every person in the dataset and return the right label accordingly.

The accuracy of each prediction is determined by a predefined confidence threshold that the predictions need to meet in order to be classed as a correct prediction. 

The Down Layer:

<img src="https://github.com/adhamarif/face-recognition/blob/main/readme_graphics/Down%20Layer.PNG">

The Network Layers:

<img src="https://github.com/adhamarif/face-recognition/blob/main/readme_graphics/Network%20Architecture.PNG">

The neural network architecture consists of the following components:

- **Down**: 6 layers performing convolution followed by maxpooling, batch normalization and ReLU resulting in a reduction in spatial dimensions.
- **Dropout**: Dropout layers with a dropout probability of 0.2 are applied after each Down layer, except for the last two layers where the dropout probability is 0.5.
- **Flatten**: A layer that flattens the output from the last convolutional layer to prepare it for fully connected layers.
- **Linear**: Fully connected layers. The first linear layer reduces the dimensionality from 8192 to 512, followed by a ReLU activation and dropout. The final linear layer reduces the dimensionality to `num_labels`.
- **ReLU**: Activation function used after the first linear layer.

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
> Remember to place the correct paths next to the `--original_path` and the `--cleaned_path` arguments.
4. Generate the **label.csv** file the data pipeline for the model training.
```
python label_generator.py --cleaned_path your_cleaned_dataset_path
```
> [!IMPORTANT]
> Remember to place the correct path next to the `--cleaned_path` argument.

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
> Remember to place the correct path next to the `--images_folder` argument.

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

> [!IMPORTANT]
> Sometimes gdown has restricted URL retrieval. If you cannot download the file automatically via gdown model, you can download the model by clicking the link provided in the error prompt. Please make sure to download the model and save it inside a folder named `models`, so that it will located in the correct path.

## Results
### Autoencoder
![image](https://github.com/adhamarif/face-recognition/assets/92054450/2937206d-3842-49c2-92f6-750956b302e4)
![image](https://github.com/adhamarif/face-recognition/assets/92054450/844c6f43-d431-4eb0-8821-38cc066c4ed8)

### Face recognition system
Results with a known face within the dataset :

<img src="https://github.com/adhamarif/face-recognition/blob/main/readme_graphics/detected.PNG" width="360" height="360"> <img src="https://github.com/adhamarif/face-recognition/blob/main/readme_graphics/AE-detected.jpg" width="360" height="360">

Results with an AI generated unknown face :

<img src="https://github.com/adhamarif/face-recognition/blob/main/readme_graphics/undetected.PNG" width="360" height="360"> <img src="https://github.com/adhamarif/face-recognition/blob/main/readme_graphics/AE-undetected.jpg" width="360" height="360">

