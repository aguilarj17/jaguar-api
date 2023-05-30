#library finction to load and run onnx model
import onnxruntime
import numpy as np
import time
#library to deal with images
from PIL import Image


def model_section(choice):
    #local variable for models directory
    models_dir = 'inferenceapi/models'
    #Conditionals to verify the version and return variables
    if choice == 1.1:
        onnx_model = "{}/JaguarPredictionV1.onnx".format(models_dir)
        tar_size=128
    elif choice == 1.2:
        onnx_model = "{}/JaguarPredictionV2.onnx".format(models_dir)
        tar_size=224
    elif choice == 2.1:
        onnx_model = "{}/ObjectDetectionV1.onnx".format(models_dir)
        tar_size=64
    elif choice == 2.2:
        onnx_model = "{}/ObjectDetectionV2.onnx".format(models_dir)
        tar_size=64
    else:
        return "invalid version"

    return onnx_model,tar_size


#function to run the inference session with onnxruntime library
def run_model(version):
    #call the function to select the model and version
    onnx_model,tar_size = model_section(version)
    #Run the module InferenceSession to load the model
    session = onnxruntime.InferenceSession(onnx_model, None)
    #get the input and ouputname of themodel
    input_name =session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return tar_size, input_name, output_name, session

def preprocess_img(img_file, tar_size):
    #Open the test image wit PIL and convert to 'RGB'
    test_img= Image.open(img_file).convert('RGB')
    #Resize the image withthe target size needed
    test_img = test_img.resize((tar_size, tar_size))
    #Transform img to array as type 'float32'
    test_img = np.array(test_img).astype('float32') / 255.0
    #Expand dims
    test_img = np.expand_dims(test_img, axis=0)
    return test_img

def run_inference(img_file, version):
    #initialize the model with onnxruntime
    tar_size, input_name,output_name,session=run_model(version)
    #call the preprocess the function and input img path and target size
    test_image= preprocess_img(img_file, tar_size)
    #start time to meassure the inference time
    start_time = time.time()
    #Run the model with the image previously trained
    output = session.run([output_name], {input_name: test_image})
    #end time
    end_time = time.time()
    #Calculate inference_time
    inference_time =end_time -start_time
    return {"Resultados": str(output),
            "inference_time": inference_time,
            "version": version}

# if __name__=="__main__":
#     jsonfile=run_inference("images/jj_0.jpg", 2.2)
#     print(jsonfile)