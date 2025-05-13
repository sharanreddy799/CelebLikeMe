import os
import pickle
import numpy as np
import cv2
import pickle
import shutil
import matplotlib.pyplot as plt

import tensorflow as tf
from skintonedetector import skintonedetector
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import Sequential, Model, load_model
import secrets

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/images'

def defaultmodel():
    vgg = load_model("./new_models/vggModel.pkl") #3 different models here
    scaler = pickle.load(open("./new_models/scaler.pkl", 'rb'))
    pca = pickle.load(open("./new_models/pca.pkl", 'rb'))
    svm = pickle.load(open("./new_models/svmModel.pkl", 'rb'))
    le = pickle.load(open("./new_models/label_encoder.pkl", 'rb'))
    return vgg,scaler,pca,svm,le
# Load your trained models here
def fairmodel():
    vgg = load_model("./Fair_models/vggModel_fair.pkl") #3 different models here
    scaler = pickle.load(open("./Fair_models/scaler_fair.pkl", 'rb'))
    pca = pickle.load(open("./Fair_models/pca_fair.pkl", 'rb'))
    svm = pickle.load(open("./Fair_models/svmModel_fair.pkl", 'rb'))
    le = pickle.load(open("./Fair_models/label_encoder_fair.pkl", 'rb'))
    return vgg,scaler,pca,svm,le
def darkmodel():
    vgg = load_model("./Dark_models/vggModel_dark.pkl") #3 different models here
    scaler = pickle.load(open("./Dark_models/scaler_dark.pkl", 'rb'))
    pca = pickle.load(open("./Dark_models/pca_dark.pkl", 'rb'))
    svm = pickle.load(open("./Dark_models/svmModel_dark.pkl", 'rb'))
    le = pickle.load(open("./Dark_models/label_encoder_dark.pkl", 'rb'))
    return vgg,scaler,pca,svm,le

def middlemodel():
    vgg = load_model("./Middle_models/vggModel_middle.pkl") #3 different models here
    scaler = pickle.load(open("./Middle_models/scaler_middle.pkl", 'rb'))
    pca = pickle.load(open("./Middle_models/pca_middle.pkl", 'rb'))
    svm = pickle.load(open("./Middle_models/svmModel_middle.pkl", 'rb'))
    le = pickle.load(open("./Middle_models/label_encoder_middle.pkl", 'rb'))
    return vgg,scaler,pca,svm,le

def modelselector(skintones):

    skintone = skintones[0]
    print('skintone at model selector:', skintone)

    match skintone:
        case 'fair': 
            vgg,scaler,pca,svm,le = fairmodel()
            print('pinged in fair')
        case 'dark':
            vgg,scaler,pca,svm,le = darkmodel()
            print('pinged in dark')
        case 'middle':
            vgg,scaler,pca,svm,le = middlemodel()
            print('pinged in middle')
        case  _:
            vgg,scaler,pca,svm,le = defaultmodel()
            print('pinged in default')
    return vgg,scaler,pca,svm,le


def get_features(path, skintone):   #send color of the person add switch case to models
    img = cv2.imread(path, 1)
    img = img[..., ::-1] 
    img = (img / 255.).astype(np.float32)  
    print('Image dimensions before resize:', img.shape)
    img = cv2.resize(img, dsize=(224, 224)) 
    #plt.imshow(img)

    print('skintone color is : ', skintone)
    vgg,scaler,pca,svm,le = modelselector(skintone)
    
    embedding_vector = vgg.predict(np.expand_dims(img, axis=0))[0]
    scaledData = scaler.transform([embedding_vector])

    pcaData = pca.transform(scaledData) 
    predictedLabel = svm.predict(pcaData)
    predictedScore = svm.predict_proba(pcaData)
    celebrityName = le.inverse_transform(predictedLabel)
    imgName = os.listdir(f"105_classes_pins_dataset/pins_{celebrityName[0]}/")[0]
    source_path = f"105_classes_pins_dataset/pins_{celebrityName[0]}/{imgName}"
    destination_dir =  os.path.join(app.config['UPLOAD_FOLDER'], "final_{}.png".format(secrets.SystemRandom().randint(1,9999)))
    shutil.copy(source_path, destination_dir)

    return celebrityName[0], max(predictedScore[0]), destination_dir


def detect_gender(image_path):
    # Load the OpenVINO model for gender and age detection
    net = cv2.dnn.readNetFromCaffe('/Users/saisharankaram/Desktop/Celebrity_LikeMe/models/deploy_gender.prototxt',
                                '/Users/saisharankaram/Desktop/Celebrity_LikeMe/models/gender_net.caffemodel')

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    


    # Define the gender classes
    gender_classes = ['Male', 'Female']

    if image is None or image.size == 0:
        print("Error: Image not found or cannot be loaded.")
         # Handle the error, e.g., exit the function or return an error response.
    else:
        # Proceed with image processing.
        blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        # Continue with the rest of the processing.

    # Set the input for the model
    net.setInput(blob)

    # Run forward pass
    gender_preds = net.forward()

    # Get the predicted gender
    gender = gender_classes[gender_preds[0].argmax()]

    return gender


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Check if the file has a valid extension (you can add more)
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Process the uploaded image file, extract features, and make predictions
            # Save the uploaded file
            
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            #save the image in a cut format
            
            print(f'Image Path is {image_path}')
            gender = detect_gender(image_path)
            skintone = skintonedetector(image_path)
            
            name, score, celebPath = get_features(image_path, skintone)
            score = '{:.2f}%'.format(score * 100)
            return render_template('index.html', celebrity_name=name, score=score, image_path=celebPath, gender = gender)
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
