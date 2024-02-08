# face-recognition
A project that implement deep learning methods for face detection. Two network are used for this project, which are **Neural Network** and **Autoencoder**. Both models served different purpose with a same goal, which to distinguish known and unknown faces. We first collected the datasets of people images and train it with these netwroks. At the end, we implement these models back into the camera system which can recognise the known faces.

## Tools used

| Library      | Version | Purpose                          |
|------------- |---------|----------------------------------|
| opencv-python| 4.8.1.78| camera & face tracking           |
| pytorch      | 2.1.0   | deep learning                    |
| matplotlib   | 3.8.2   | image plotting                   |
| pandas       | 2.1.3   | metadata management              |
| numpy        | 1.24.1  | mathematical operations on array |

Other dependencies are listed as well in [requirements.txt](requirements.txt)

## Installation

1 Clone this repository  
```
git clone https://github.com/adhamarif/face-recognition   
```

2. Make sure you are on the correct directory.
```
cd face-recognition  
```

3. Install the dependencies:   
```
pip install -r requirements.txt  
```

4. If you would like to train the network using graphical card, we encouraged you to download the [Pytorch](https://pytorch.org/get-started/locally/) library from its website.

## Network architecture
2 different model architecture are used to execute 2 different tasks for face recognition

### Autoencoder
Datasets of known face are trained by using **Autoencoder** to reconstruct the image. At the end, if the unknown face is given as the input, the autoencoder will not be able to reconstruct the image and will resulted in huge loss value.

The loss value (of validation set) will then be taken as threshold value in order to distinguish known and unknown faces.

[insert AE model architecture]

### Neural Network
Datasets of known face are trained by **Neural Network** to learn the features in every images and at the end make a prediction of every person in the dataset. This network needs a label to make a prediction.

[explaination on how to determine the threshold]

[insert Neural Network architecture]

## Usage

### Use the face recognition system with a pre-trained model.
We have uploaded our checkpoints in the google drive and you can execute the face recognition system with the pre-trained model without the need to train it again from scratch.

Autoencoder:
```
python main.py --model autoencoder
```
Neural Network:
```
python main.py --model network
```

### Train from scratch
If you wish to train the model again from scratch with your own datasets, you might need to prepare your own datasets first before you can start train the model.

#### Dataset preparation
1. Capture the face of known face and save it under the same (label) folder by running:
```
python collect_data.py
```

2. Check the datasets in the face folder. If you are fine with the collected face images, you may proceed to step 4.
3. If the collected (face) images have a lot of noises or undesired object, you might need to either:
- recollect the images dataset, **OR**
- run the cleaning dataset script to get all the (face) images detected by the *haarcascades* algorithm. After that, you may need to remove the noise and undesired images from the <mark style="background-color: #2ec4b6">NEW_FOLDER</mark> manually.
```
python clean_dataset.py
```
4. Generate the **label.csv** file the data pipeline for the model training.
```
python label_generator.py
```

! Please make sure the PATH in the script is adapted with your desired PATH.

#### Train the model
Autoencoder:
```
python train_autoencoder.py
```

Neural Network:


[insert train neural network file here]

! Please make sure that you check the PATH for the  <mark style="background-color: #2ec4b6">LABEL_FILE</mark> and  <mark style="background-color: #2ec4b6">IMAGE_FOLDER</mark> are adapted with your desired PATH.

#### Run the facial recognition system with the new model
1. Make sure the new model is available on your local (main) directory. Otherwise, it will still fetch the pre-trained model from the google drive
2. Run the facial recognition system

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