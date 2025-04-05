import sys
import os
sys.path.append(os.getcwd() + "/fast_api_inference")

from fastapi import FastAPI
from fastapi import UploadFile
from keras.models import load_model # type: ignore
from face_detector import face_detector
from process_pretrain_network import resnet_50
from configs import configs




app = FastAPI()

model = load_model(configs.MODEL_PATH)

@app.get("/")
def root():
    return {"response":"this api is working on root"}

@app.post("/video")
def detect(file : UploadFile):

    path = face_detector.video_save(file)

    array = face_detector.video_to_numpy(path)
    array = resnet_50.resnet_50(array)
    result = model.predict(array, verbose = False)

    
    return {"{Probability of being fake}" : 1 - float(result[0])}