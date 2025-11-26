# Sign Language Translator

A real-time, two-handed sign language translator built with Python, MediaPipe, and Scikit-learn. This project captures hand gestures via webcam, extracts landmarks, and translates them into text using a Random Forest classifier. It includes a Flask-based web application for easy usage.

## Features

-   **Real-time Recognition**: Translates gestures instantly using a lightweight Random Forest model.
-   **2-Hand Support**: Detects and processes landmarks from both left and right hands (126 features).
-   **Web Interface**: Simple, browser-based UI powered by Flask and OpenCV video streaming.
-   **Custom Dataset**: Easy-to-use data collector to create your own sign language dictionary.

## Tech Stack

-   **Python 3.12+**
-   **MediaPipe**: For robust hand tracking and landmark extraction.
-   **Scikit-learn**: For the Random Forest classifier.
-   **OpenCV**: For video capture and image processing.
-   **Flask**: For the web application backend.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sign-language-translator.git
    cd sign-language-translator
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Collect Data (Optional)
If you want to train the model on your own gestures:
```bash
python src/data_collector.py
```
-   Type a label (e.g., "Hello") and press **ENTER**.
-   Press **SPACE** to start/stop recording (aim for ~50-100 frames per word).
-   Press **ENTER** again to switch to a new word.

### 2. Train the Model
Once you have collected data, train the classifier:
```bash
python src/train_classifier.py
```
This will save the trained model to `models/isl_classifier.p`.

### 3. Run the Web App
Start the translator:
```bash
python app/main.py
```
Open your browser and go to `http://localhost:5000`.

## Project Structure

-   `app/`: Flask web application code.
-   `src/`: Core scripts for data collection, training, and inference.
-   `data/`: Stores the collected landmark dataset (`landmarks_data.csv`).
-   `models/`: Stores the trained model (`isl_classifier.p`).

## License

MIT
