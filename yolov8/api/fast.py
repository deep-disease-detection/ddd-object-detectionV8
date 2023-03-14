from fastapi import FastAPI
from ultralytics import YOLO


app = FastAPI()
app.state.model = YOLO('../best.pt')


@app.post('/detect/')
def detect(image_path):
    results = app.state.model(image_path)
    #fonction de youssef pour plot
