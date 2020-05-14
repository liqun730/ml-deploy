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

@app.get("/")
async def main():
    content = """
        <body>
        <form action="/file/" enctype="multipart/form-data" method="post">
        <input name="file" type="file" multiple>
        <input type="submit">
        </body>
    """
    return HTMLResponse(content=content)
