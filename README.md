# Handwritten Digit Classifier

This is a desktop application that uses a deep learning model to recognize handwritten digits from images.  
Built with Python, TensorFlow, and Kivy for a simple and interactive user interface.

## What It Does

- Lets you upload an image of a handwritten digit.
- Uses a trained neural network to predict the digit.
- Displays the prediction along with the confidence level.

## How to Use

1. **Download the project.**
2. **Download the trained model (`mnist_transfer_model.h5`)** from my Kaggle project:  
   link to my [Kaggle project](https://www.kaggle.com/code/zlatan599/handwritten-digit-recognition).
3. Place the `mnist_transfer_model.h5` file inside the `models/` folder in the project directory.
4. If you want, choose a different logo immage (`LOGO.png`) and put it inside `assets/`.
5. Run the Python file (`GUI.py`).
6. Click "Upload Image" and select a photo of a handwritten digit.

## Notes

- The model is not included here because GitHub does not allow files over 25 MB.
- Make sure to install Python and required libraries (e.g., TensorFlow, Kivy) before running.
- Works on Windows, macOS, and Linux.

---

Created entirely by Leonardo Cofone.
