CNN Image Classifier – Cat vs Dog
This is a Convolutional Neural Network (CNN) based binary image classification project built using Python and TensorFlow/Keras. The model is trained to classify whether an input image is of a cat or a dog using image data preprocessing and deep learning.

Features
CNN model from scratch using Keras Sequential API

Image preprocessing using ImageDataGenerator

Train-validation split with data augmentation

Accuracy/loss visualization using matplotlib

Predict image class (Cat/Dog) using model.predict()

Save and load trained model (model.h5)

Tech Stack
Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

PIL (for image loading and prediction)

Folder Structure
bash
Copy
Edit
cat-dog-cnn/
├── cnn.py              # Model training and prediction script
├── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   └── dogs/
│   └── test_set/
│       ├── cats/
│       └── dogs/
├── requirements.txt
└── README.md
How It Works
The dataset is loaded using ImageDataGenerator and resized to 64x64.

The model is trained with augmented images (shear, zoom, flip).

After training, it is saved as model.h5.

A separate script (predict.py) is used to predict new unseen images.

Getting Started
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/cat-dog-cnn.git
cd cat-dog-cnn
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train the model
bash
Copy
Edit
python cnn.py
This will train the model and save it as model.h5.

4. Predict on a new image
Place the image you want to predict (e.g., sample_image.jpg) in the root folder and run:

bash
Copy
Edit
python predict.py
The script will print whether it is a Cat or a Dog.

Sample Output
bash
Copy
Edit
[INFO] Loading image...
[INFO] Predicted: Dog
Model Overview
Input Shape: (64, 64, 3)

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Data Augmentation: Enabled (shear, zoom, flip)

Training Visualization
Includes plots for:

Accuracy vs Epochs

Loss vs Epochs

These are saved/shown using matplotlib.

License
This project is licensed under the MIT License
