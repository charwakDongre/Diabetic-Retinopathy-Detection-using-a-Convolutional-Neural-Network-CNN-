# Diabetic Retinopathy Detection using a CNN

This project demonstrates the use of a Convolutional Neural Network (CNN) for multi-class image classification. The goal is to build and deploy a deep learning model that can accurately classify retinal fundus images into one of five stages of Diabetic Retinopathy (DR). The analysis covers data preprocessing and augmentation, building a custom CNN, training the model, and deploying it with a Flask web application.

---

## The Dataset

The project uses a private dataset of retinal fundus images organized into five classes corresponding to the severity of Diabetic Retinopathy (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative). The data is pre-split into `train`, `val`, and `test` directories for proper model training and evaluation.

---

## Project Workflow

### 1. Data Preprocessing & Augmentation
- **Image Loading**: The `ImageDataGenerator` from Keras was used to efficiently load images from the directories.
- **Normalization**: Pixel values for all images were rescaled from the [0, 255] range to [0, 1], a critical step for stabilizing the training process.
- **Data Augmentation**: To prevent overfitting and improve generalization, the training dataset was augmented with random transformations, including rotations, zooms, and horizontal flips.

### 2. CNN Model Architecture
A custom `Sequential` CNN model was built using TensorFlow and Keras. The architecture consists of:
- **Convolutional Layers**: Two `Conv2D` layers with `relu` activation were used to extract hierarchical features from the images.
- **Pooling Layers**: `MaxPooling2D` layers were used to downsample the feature maps, reducing computational complexity and making the model more robust.
- **Dense Layers**: After flattening the feature maps, two `Dense` layers were used for classification, with a `Dropout` layer included to prevent overfitting.
- **Output Layer**: A final `Dense` layer with a `softmax` activation function outputs a probability distribution over the five DR classes.

### 3. Model Training & Evaluation
The model was trained for a set number of epochs, with its performance monitored on the validation set.
- **Training**: The model was trained using the `adam` optimizer and `categorical_crossentropy` loss function, which are standard choices for multi-class classification.
- **Evaluation**: After training, the model's final performance was measured on the unseen test set to get an unbiased estimate of its accuracy in a real-world scenario.
- **Visualization**: The training and validation accuracy/loss were plotted to visualize the model's learning progress and check for overfitting.

### 4. Web Application Deployment
The trained model (`re_model.h5`) was deployed using a Flask web application.
- **Backend**: The Flask server (`app.py`) loads the saved Keras model and handles file uploads.
- **Frontend**: A simple HTML/CSS interface allows users to upload a retinal image.
- **Prediction**: The application preprocesses the uploaded image, feeds it to the model, and displays the predicted DR stage along with the model's confidence score.

---

## How to Run the Code
1.  Ensure you have Python installed.
2.  Clone this repository to your local machine.
3.  Install the required libraries:
    ```bash
    pip install tensorflow flask numpy matplotlib
    ```
4.  Place your dataset into `train`, `val`, and `test` folders inside a main `dataset` directory.
5.  Run the training script to generate the model file (optional, if you don't have one):
    ```bash
    python train_model.py
    ```
6.  Run the Flask application to start the web server:
    ```bash
    python app.py
    ```
7.  Open your browser and navigate to `http://127.0.0.1:5000`.

---

## Libraries Used
- **TensorFlow**
- **Flask**
- **numpy**
- **matplotlib**
