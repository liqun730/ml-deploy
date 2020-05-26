import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from app.classifier import Classifier

model = Classifier('./data/classfier.h5')

app = FastAPI()

@app.post('/predict')
def predict(image: UploadFile = File(...)):
    # predict label
    return model.predict_from_file(image.file)
