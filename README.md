# Fashion MNIST Image Classification
<img src="images/unsplash.jpg" alt="unsplash" width="949" height="400">(Photo by <a href="https://unsplash.com/@the_modern_life_mrs?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Heather Ford</a> on <a href="https://unsplash.com/photos/5gkYsrH_ebY?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
  )

#### **Author**:[Eugene Kuloba](https://github.com/eugenekuloba)

<p>This project is an image classification task using the Fashion MNIST dataset. Fashion MNIST is a dataset of 28x28 grayscale images of 10 different fashion categories, including items like T-shirts, dresses, sneakers, and more. The goal of this project is to build a convolutional neural network (CNN) model that can accurately classify these fashion items.</p>

## Key Features:

1. Data Preprocessing: The dataset is loaded, preprocessed, and prepared for training and testing.
2. CNN Model Architecture: A Convolutional Neural Network (CNN) model is defined to learn and recognize patterns in the images.
3. Model Training: The model is trained on the training dataset using TensorFlow/Keras.
4. Evaluation: The model's performance is evaluated on a test dataset, including metrics such as accuracy and loss.
5. Results: The project reports the achieved test accuracy and loss, showcasing the model's effectiveness.


## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)


## Installation

To run this project locally, follow these steps:

```bash
# clone repository
git clone https://github.com/username/Fashion-MNIST
cd Fashion-MNIST

# Create a virtual environment (Python 3.x)
python3 -m venv venv

# Activate the virtual environment (Linux/macOS)
source venv/bin/activate

# Activate the virtual environment (Windows)
venv\Scripts\activate

# install required modules
pip install -r requirements.txt
```

## Dataset

This project uses the Fashion MNIST dataset, which is a dataset of grayscale images depicting 10 different fashion categories. Each image is 28x28 pixels in size. The dataset is commonly used as a benchmark for image classification tasks and is a more challenging alternative to the classic MNIST dataset.

**Dataset Details:**

- Number of Classes: 10
- Categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Training Samples: 60,000
- Test Samples: 10,000
- Image Size: 28x28 pixels

**Dataset Source:**

The Fashion MNIST dataset is readily available within the Keras library and can be loaded using the `fashion_mnist.load_data()` function.

**Data Preprocessing:**

Before training the model, the dataset is preprocessed as follows:

- Reshaped: Images are reshaped to have a single channel (grayscale) and dimensions of (28, 28, 1).
- Normalized: Pixel values are normalized to the range [0, 1] by dividing by 255.0.
- One-Hot Encoding: Labels are one-hot encoded to convert them into a format suitable for multi-class classification.

**Usage:**

This dataset is an excellent resource for experimenting with image classification and deep learning models. It can be used for educational purposes or as a benchmark dataset for evaluating different machine learning algorithms.

If you'd like to explore the dataset and use it in your own projects, you can easily access it via the Keras library or download it from the official Fashion MNIST repository.

For more details about the dataset, refer to the official Fashion MNIST documentation.


## Model Architecture

The convolutional neural network (CNN) model used in this project is designed to perform image classification on the Fashion MNIST dataset. CNNs are well-suited for image recognition tasks due to their ability to capture hierarchical features in images.

**Model Overview:**

The CNN model consists of several layers, including convolutional layers, max-pooling layers, and fully connected layers. Here's an overview of the model architecture:

1. **Input Layer:**
   - Input Shape: (28, 28, 1)
   - This layer accepts 28x28-pixel grayscale images as input.

2. **Convolutional Layers:**
   - Two sets of convolutional layers with 32 filters each.
   - Each convolutional layer uses a 3x3 kernel size.
   - Activation Function: ReLU (Rectified Linear Unit)
   - These layers extract features from the input images.

3. **Max-Pooling Layers:**
   - Two max-pooling layers follow the convolutional layers.
   - Each max-pooling layer uses a 2x2 pooling window.
   - Max-pooling reduces spatial dimensions, aiding in feature extraction.

4. **Convolutional Layers (Second Set):**
   - Another two sets of convolutional layers with 64 filters each.
   - Each convolutional layer uses a 3x3 kernel size.
   - Activation Function: ReLU
   - These layers further refine the learned features.

5. **Max-Pooling Layers (Second Set):**
   - Two additional max-pooling layers follow the second set of convolutional layers.
   - Each max-pooling layer uses a 2x2 pooling window.

6. **Flatten Layer:**
   - The output from the convolutional layers is flattened into a 1D vector.
   - This prepares the data for the fully connected layers.

7. **Fully Connected Layers:**
   - A dense layer with 128 units and ReLU activation function.
   - This layer learns high-level representations from the extracted features.

8. **Output Layer:**
   - A dense layer with 10 units (corresponding to the 10 fashion categories in the dataset).
   - Activation Function: Softmax
   - The softmax activation function produces class probabilities.

**Training Configuration:**

The model is trained using the Adam optimizer and categorical cross-entropy loss function. The training process involves forward and backward passes through the network to optimize the model's weights.

The final trained model achieves a test accuracy of approximately 92.54%, indicating its effectiveness in classifying Fashion MNIST images.

This model architecture serves as a starting point for image classification tasks and can be further customized or extended to suit specific requirements.

## Training

Instructions on how to train the model, including hyperparameters and training duration.

## Evaluation

<p> After running the model and doing my model evaluation, I got the following results.

1. Test Loss (0.276): This represents the average loss (error) on the test dataset. Lower values are better, indicating that the model's predictions are closer to the true labels.

2. Test Accuracy (92.54%): This is the proportion of correctly classified samples in the test dataset. A higher accuracy indicates that the model is making accurate predictions.




