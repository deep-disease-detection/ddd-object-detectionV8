from fastapi import FastAPI
from ultralytics import YOLO
import os
from pydantic import BaseModel
import base64
from imageio import imread
import io
import numpy as np
import cv2
from yolov8.api.utils import save_image_to_bucket_gcp
from yolov8.params import *


app = FastAPI()
app.state.model = YOLO(os.environ.get("PATH_TO_MODEL"))


@app.get("/")
def hello():
    return {"hello": "marche"}


class Item(BaseModel):
    image: str


@app.post("/predict/")
def detect(image_json: Item):

    b64code = image_json.image
    b = base64.b64decode(b64code)
    img = imread(io.BytesIO(b))

    # if there are only two channels in the image
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)

    result = app.state.model(img)  # prediction of our model

    res = result[0]  # take first element (only one image for api)

    # no viruses were found by YOLOv8
    if len(res.boxes) == 0:
        img_bytes = cv2.imencode(".jpeg", img)[1].tostring()
        save_image_to_bucket_gcp(img_bytes, os.environ.get("BUCKET"), "no virus", 0)
        return {"virus": "no virus found", "virus_count": 0}

    # keep only one class to avoid predicting on noise
    predicted_classes_scores = {c: 0 for c in res.boxes.cls.numpy().tolist()}

    for cls, conf in zip(res.boxes.cls, res.boxes.conf):
        predicted_classes_scores[float(cls)] += float(conf)

    predicted_class = int(
        list(predicted_classes_scores.keys())[
            np.argmax(predicted_classes_scores.values())
        ]
    )
    predicted_virus = VIRUSES[predicted_class]
    final_boxes = [box for box in res.boxes if int(box.cls) == predicted_class]
    predicted_virus_count = len(final_boxes)

    # PARAMETERS FOR DISPLAY
    MARGIN = 10
    FONT_THICKNESS = 2
    BOX_THICKNESS = 4
    COLOR = (112, 224, 0)
    FONT_SCALE = 2
    FONT = cv2.FONT_HERSHEY_PLAIN

    # Add boxes and text to to image
    for box in final_boxes:
        img = cv2.rectangle(
            img,
            (int(box.xyxy[0, 0]), int(box.xyxy[0, 1])),
            (int(box.xyxy[0, 2]), int(box.xyxy[0, 3])),
            color=COLOR,
            thickness=BOX_THICKNESS,
        )
        org = (
            int(box.xyxy[0, 2] - box.xywh[0, 2]),
            int(box.xyxy[0, 3] - box.xywh[0, 3] - MARGIN / 2),
        )
        img = cv2.putText(
            img,
            text=f"{round(float(box.conf),2)}",
            org=org,
            fontFace=FONT,
            fontScale=FONT_SCALE,
            color=COLOR,
            thickness=FONT_THICKNESS,
        )

    img_bytes = cv2.imencode(".jpeg", img)[1].tostring()

    save_image_to_bucket_gcp(
        img_bytes, os.environ.get("BUCKET"), predicted_virus, predicted_virus_count
    )

    return {"virus": predicted_virus, "virus_count": predicted_virus_count}
