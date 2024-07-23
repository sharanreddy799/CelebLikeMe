# Celebrity LikeMe

Celebrity LikeMe is a web application that utilizes machine learning models to detect skin tone, recognize faces, and predict the closest celebrity match based on facial features.

## 1 .Features

- **Skin Tone Detection:** Determines the skin tone of a person in an image using the Skin Tone Detector library.
- **Gender Detection:** Utilizes OpenVINO models to detect the gender of a person in the uploaded image.
- **Face Recognition:** Identifies faces in an image, extracts facial features, and predicts the closest celebrity match based on pre-trained models.


## 2. Requirements

* Download the dataset from kagge using read me in the dataset.

* make a copy of it and put it into labelled folder and seperate them based on the given folder structure

* Download face weights using read me in the face weights folder

* Download Shape predictor using read me into the shape predictor folder.

* Install latest Python

## 3. Installation

Install the required dependencies:

```python
pip install -r requirements.txt
```
Download additional model files as specified in the code and update file paths accordingly.


## 4. Training Models

The application uses various models for skin tone detection, gender detection, and face recognition. These models are specified in the code and should be downloaded and configured accordingly.

Download the data from kaggle and divide them into their directories based on the directories.txt


## 5. Run the Flask web application:
```python
python your_app_file.py
````
Open your web browser and navigate to http://localhost:5000.

Upload an image using the provided form and click the "Predict" button.

View the predicted celebrity match, confidence score, and the uploaded image with the celebrity.




