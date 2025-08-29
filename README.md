# Handwritten Digit Recognition

A Pygame-based application for recognizing handwritten digits using a pre-trained TensorFlow/Keras model.

## Features

- Real-time digit drawing interface
- MNIST digit recognition using deep learning
- Visual feedback of predictions
- Debug window showing what the model "sees"
- Top 3 prediction probabilities display
- Board clearing functionality

## Requirements

- Python 3.7+
- Pygame
- TensorFlow/Keras
- OpenCV
- NumPy

## Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install pygame tensorflow opencv-python numpy
```

## Usage

1. Run the application:
```bash
python ui/Handwritten_digit_uii.py
```

2. Draw a digit (0-9) using your mouse
3. Release the mouse button to get the prediction
4. The predicted digit will be displayed on the screen
5. Press 'n' key to clear the drawing board
6. A debug window will show the preprocessed image that the model sees

## How it Works

### Image Preprocessing Pipeline

The application uses an optimized preprocessing pipeline:

1. **Color Inversion**: Converts white-on-black drawing to black-on-white (MNIST format)
2. **Thresholding**: Cleans up the image using binary thresholding
3. **Aspect Ratio Preservation**: Adds padding to maintain digit proportions
4. **Single Resize**: Resizes to 28x28 pixels (MNIST input size)
5. **Normalization**: Scales pixel values to [0, 1] range

### Model

The application uses a pre-trained convolutional neural network (CNN) trained on the MNIST dataset. The model is loaded from `model_training/model.keras`.

## Troubleshooting

### Digit "1" Recognition Issues

If digit "1" is not being recognized correctly:

1. **Draw clearly**: Make sure the digit is drawn with a single, clear vertical line
2. **Check debug window**: Verify the preprocessed image shows a clear "1" shape
3. **Adjust drawing style**: Try drawing the digit with slightly thicker strokes
4. **Model training**: The model may need retraining with more "1" examples

### Common Issues

- **Double predictions**: Clear the board with 'n' key between drawings
- **Poor recognition**: Ensure good lighting and clear drawing
- **Debug window not showing**: Check if OpenCV windows are enabled on your system

## File Structure

```
mnist_digit_recognition/
├── ui/
│   └── Handwritten_digit_uii.py  # Main application
├── model_training/
│   ├── model_training_script.ipynb  # Training notebook
│   └── model.keras  # Pre-trained model
├── assets/
│   ├── fonts/  # Font files
│   └── Images/  # Image assets
└── README.md  # This file
```

## Controls

- **Mouse**: Draw digits by clicking and dragging
- **'n' key**: Clear the drawing board
- **ESC or Close window**: Exit the application

## Model Performance

The model provides:
- Primary prediction with confidence score
- Top 3 predictions for better insight
- Debug visualization of input image

## Contributing

Feel free to contribute by:
- Improving the preprocessing pipeline
- Adding more training data
- Enhancing the UI/UX
- Adding new features

## License

This project is for educational purposes. Feel free to use and modify as needed.
