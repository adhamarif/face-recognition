# face-recognition
A project that implement deep learning methods for face detection. Two network are used for this project, which are **classical neural network** and **autoencoder**. Both models served different purpose with a same goal, which to distinguish known and unknown faces. We first collected the datasets of people images and train it with these netwroks. At the end, we implement these models back into the camera system which can recognise the known faces.

## Tools used

| Library      | Version | Purpose                          |
|------------- |---------|----------------------------------|
| opencv-python| 4.8.1.78| camera & face tracking           |
| pytorch      | 2.1.0   | deep learning                    |
| matplotlib   | 3.8.2   | image plotting                   |
| pandas       | 2.1.3   | metadata management              |
| numpy        | 1.24.1  | mathematical operations on array |

## How to use

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

## Results
