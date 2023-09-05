# lets begin boiii
import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

from utils.fertilizers import fertilizer_dict
from utils.disease import disease_dic
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template



disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']




with open('YieldPredModelXGB.pkl', 'rb') as file:
    YieldPredModel= pickle.load(file)

with open('District.pkl', 'rb') as dict_file:
    District_dict = pickle.load(dict_file)
# print(District_dict)

with open('Crop.pkl', 'rb') as dict_file:
    Crop_dict = pickle.load(dict_file)
# print(Crop_dict)
# print(Crop_dict['Wheat'])

with open('Season.pkl', 'rb') as dict_file:
    Season_dict = pickle.load(dict_file)
# print(Season_dict)

with open('Year.pkl', 'rb') as dict_file:
    Year_dict = pickle.load(dict_file)
# print(Year_dict)


def yield_predict():
    print(YieldPredModel.predict(
        [[District_dict['NICOBARS'], Crop_dict['Arecanut'], Year_dict['2001'], Season_dict['Kharif'], 1254]]))

# yield_predict()




MODEL_PATH ='plantsmodel.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print("model read")

# img_path= 'Tomato___Septoria_leaf_spot.JPG'


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    print("image loaded ")

    # Preprocessing the image
    x = image.img_to_array(img)


    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    print("image supplied to model")
    preds = model.predict(x)
    print(" model prediction done")
    preds = np.argmax(preds, axis=1)
    prediction=disease_classes[preds[0]]



    print (preds)
    return render_template('disease_detection_result.html', recommendation=disease_dic[prediction])


def fertilizer_Rec():
    pd.read_csv('fertilizer.csv')



#
# model_predict(img_path, model)


app = Flask(__name__, static_url_path='/static')


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index_backup.html')


#-------------------------------------------------------------------------------Yield Prediction ---------------------------------------------------------------------------------

@app.route('/crop_yield.html', methods=['GET', 'POST'])
def crop_yield():
    if request.method == 'POST':
         state = request.form['state']
         district = request.form['district']
         crop = request.form['crop']
         year = '2019'
         season = request.form['season']
         area = float(request.form['area'])
         print( state , district , crop , year , season , area)

         inputs = ( District_dict[district.upper()], Crop_dict[crop], Year_dict[year], Season_dict[season], area*100)

         input_data_as_numpy_array = np.asarray(inputs)

         input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

         prediction = YieldPredModel.predict(input_data_reshaped)

         print(str(prediction[0])[:3])

         return render_template('crop_yield_result.html', recommendation=str(prediction[0])[:3])

    return render_template('crop_yield.html')

## -------------------------------------------------- CROP RECOMMEMDATION-------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

with open('Crop_dict.pkl', 'rb') as dict_file:
    Crop_dict_rec = pickle.load(dict_file)
print(Crop_dict_rec)


with open('CropRec_model.pkl', 'rb') as file:
    CropRec_model = pickle.load(file)

with open('scalerCR.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

# 90	42	43	20.879744	82.002744	6.502985	202.935536
scaled_input=sc.transform([[90,42,43,20.879744,82.002744,6.502985,202.935536]])
preds=CropRec_model.predict(scaled_input)

@app.route('/crop_recommendation.html', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        print('post succesful')
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        scaled_input = sc.transform([[nitrogen, phosphorus,potassium, temperature,humidity,ph,rainfall]])
        print(scaled_input)
        preds = CropRec_model.predict(scaled_input)
        print(preds)
        return render_template('crop_recommendation_result.html', recommendation=Crop_dict_rec[preds[0]])


    return render_template('crop_recommendation.html')

#-----------------------------------------------------FERTILIZER RECOMMENDATION-----------------------------------------------------------------------------------------------------
@app.route('/fertilizer_recommendation.html')
def fertilizer_recommendation():
    return render_template('fertilizer_recommendation.html')

@app.route('/recommend', methods=['POST'])
def recommend_fertilizer():
    crop_name = request.form['crop']
    N = float(request.form['nitrogen'])
    P = float(request.form['phosphorous'])
    K = float(request.form['potassium'])
    ph_value = float(request.form['phValue'])
    moisture = float(request.form['moisture'])

    df = pd.read_csv('fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    # print(fertilizer_dict[key])
    return render_template('recommendation_result.html', recommendation=fertilizer_dict[key])



@app.route('/plant_disease_detection.html', methods=['GET', 'POST'])
def plant_disease_detection():

    if request.method == 'POST':
        # Check if the post request has a file part
        print(" Post succesful")
        if 'image' not in request.files:
            return "No file part"

        image_file = request.files['image']

        # If the user does not select a file, the browser submits an empty part without filename
        if image_file.filename == '':
            return "No selected file"

        # Save the uploaded image to the specified folder
        if image_file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(filename)
            print("printing filename")
            print(filename)
            preds = model_predict(filename,model)
            return preds

    return render_template('plant_disease_detection.html')



app.run(host="0.0.0.0", port=80, debug=True)

print("Done ")