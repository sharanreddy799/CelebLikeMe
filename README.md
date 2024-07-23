# Celebrity LikeMe

Celebrity LikeMe is a web application that utilizes machine learning models to detect skin tone, recognize faces, and predict the closest celebrity match based on facial features.

## Features

- **Skin Tone Detection:** Determines the skin tone of a person in an image using the Skin Tone Detector library.
- **Gender Detection:** Utilizes OpenVINO models to detect the gender of a person in the uploaded image.

- **Face Recognition:** Identifies faces in an image, extracts facial features, and predicts the closest celebrity match based on pre-trained models.

## Installation

Install the required dependencies:

pip install -r requirements.txt
Download additional model files as specified in the code and update file paths accordingly.
Usage

## Run the Flask web application:

python your_app_file.py
Open your web browser and navigate to http://localhost:5000.
Upload an image using the provided form and click the "Predict" button.
View the predicted celebrity match, confidence score, and the uploaded image with the celebrity.

## Models

The application uses various models for skin tone detection, gender detection, and face recognition. These models are specified in the code and should be downloaded and configured accordingly.
