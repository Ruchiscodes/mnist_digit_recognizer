# mnist_digit_recognizer
# Digit Recognizer

This project implements a deep learning model to recognize handwritten digits using the MNIST dataset. It includes an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) for improved accuracy.

## Features
- **ANN Model:** Implemented from scratch using NumPy and Pandas, achieving **93.71% accuracy**.
- **CNN Model:** Uses two convolutional layers followed by max pooling, achieving **99.19% accuracy**.
- **Interactive GUI:** A simple interface to draw digits and get real-time predictions.

## Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/Digit-Recognizer.git
   cd Digit-Recognizer
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```sh
   jupyter notebook mnist_digit_recognizer.ipynb
   ```

## Usage
- Train the model using `mnist_digit_recognizer.ipynb`.
- Save the trained model as `mnist_cnn_model.h5`.
- Use the GUI to draw digits and test predictions in real-time.

## Model Architecture
### **Convolutional Neural Network (CNN)**
- **Conv2D (32 filters, 3x3 kernel, ReLU activation)**
- **MaxPooling2D (2x2 pool size)**
- **Conv2D (64 filters, 3x3 kernel, ReLU activation)**
- **MaxPooling2D (2x2 pool size)**
- **Flatten Layer**
- **Dense (128 neurons, ReLU activation)**
- **Dense (10 neurons, Softmax activation for classification)**

## Results
| Model | Accuracy |
|--------|----------|
| ANN (NumPy) | 93.71% |
| CNN (TensorFlow) | 99.19% |

## Future Improvements
- Implement more data augmentation techniques to improve generalization.
- Optimize model performance using transfer learning.
- Improve the user interface for better interaction.

