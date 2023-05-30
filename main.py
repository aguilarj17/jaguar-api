from inferenceapi import create_app
from inferenceapi.model_pred import run_inference
from flask import request,jsonify, make_response, redirect, session
import json
import os

#Initialize the app
app = create_app()

#Golbal variables
IMG_DIR ='inferenceapi/images/'

#models db
models_db =[{"onnx model name": "wHC_1.onnx",
             "id_version":1.1},
            {"onnx model name": "wHC_1_1.onnx",
             "id_version": 1.2},
            {"onnx model name": "model_bbox_regression_and_classification_m1_vf.onnx",
             "id_version": 2.1},
            {"onnx model name": "wHC_1.onnx",
             "id_version": 2.2},
    ]
#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = set(['jpg' , 'jpeg' , 'png'])

#Function to verify if the files are in ALLOWED_EXTENSIONS
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    user_ip= request.remote_addr
    response=make_response(redirect('/models'))
    #response.set_cookie('user_ip',user_ip)
    session['user_ip']= user_ip
    return response


#route to show the models_db
@app.route('/models')
def wecome():
    return jsonify(models_db), 200


@app.route('/models/prediction', methods = ["GET","POST"])
def jaguar_prediction():
    if request.method=="POST":
        #request the json data
        img_data=json.loads(request.data)
        #Error checking if there exist a file and if the filename is an empty string
        if img_data is None or img_data["filename"]=="":
            #if not exist a file the it will return an error
            return jsonify({"error": "Empty-data"})
        else:
            if allowed_file(img_data["filename"]):
                #joint the filename with the IMG_DIR
                file_path = os.path.join(IMG_DIR, img_data["filename"])
                #call the fucntion to run the inference with onnxruntime
                output=run_inference(file_path, 1.1)
                #print(jsonfile)
                return jsonify({"filename": img_data["filename"],
                                "outputs":output})
            else:
                #return error to show that the file are not allowed
                return jsonify({"error": "The chossen file is not allowed"})               

    else:
        #return a error if a request has a  method different of "POST"
        return jsonify({"error": "The HTTP protocol only allows Post methods"})

    return "Inference API"
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')