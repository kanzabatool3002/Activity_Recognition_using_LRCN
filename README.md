# Video Activity Recognition using LRCN (Long-term Recurrent Convolutional Network)

## Why LRCN?
LRCN (Long-term Recurrent Convolutional Network) is a deep learning architecture that combines **CNNs (Convolutional Neural Networks)** for spatial feature extraction and **LSTMs (Long Short-Term Memory networks)** for temporal sequence modeling. This makes it well-suited for video-based activity recognition, where both spatial and temporal information are crucial.

### Key Benefits of Using LRCN for Video Activity Recognition:
- **Capturing Spatial Features:** CNN layers extract meaningful spatial information from each video frame.
- **Understanding Temporal Dynamics:** LSTM layers capture sequential dependencies between frames, improving activity recognition.
- **Better than Frame-Wise CNNs:** Unlike simple CNN-based classifiers that process frames independently, LRCN maintains context over time.
- **Handling Variable-Length Videos:** LSTMs allow processing videos of different durations while preserving sequential relationships.

## Dataset & Preprocessing:
The dataset, stored on **Google Drive**, consists of labeled video sequences. These videos undergo preprocessing steps, including:
- Extracting frames at equal intervals.
- Resizing frames to **64×64 pixels**.
- Normalizing pixel values to the range **[0,1]**.
- Preparing data sequences for model training.

Training was conducted on **Google Colab**, using Python libraries including:
- **TensorFlow** for deep learning.
- **OpenCV & MoviePy** for video processing.
- **Matplotlib** for visualization.
- **Scikit-learn** for dataset splitting.

## LRCN Model Architecture:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kanzabatool3002/Activity_Recognition_using_LRCN/blob/main/Activity_Recognition_Using__LRCN.ipynb)


The **LRCN** model consists of:
- **TimeDistributed Conv2D layers** with ReLU activation for spatial feature extraction.
- **MaxPooling2D layers** for spatial downsampling.
- **Dropout layers** to prevent overfitting.
- **Flatten layer** to convert spatial features into a 1D vector.
- **LSTM layer** to capture temporal dependencies in the video sequence.
- **Dense layer with softmax activation** for multi-class classification.


## Training & Optimization:
The model was trained using the **Adam optimizer** with:
- Learning rate: **0.0001**
- Loss function: **Categorical Crossentropy**
- Evaluation metric: **Accuracy**

### Training Configuration:
- **70 epochs** with batch size **16**.
- **EarlyStopping** (stops training if validation loss doesn’t improve for **15 epochs**).
- **Data Split:** 80% for training, 20% for validation.

## Performance & Evaluation:
After extensive training and testing, the **LRCN model** achieves an **accuracy of 68%**, demonstrating its ability to recognize human activities from video sequences.

## To test the trained model, refer to this repository:

[Testing Activity Recognition Model](https://github.com/kanzabatool3002/Activity_Recognition_Model_Testing.git)
