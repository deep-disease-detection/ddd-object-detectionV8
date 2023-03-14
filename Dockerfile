FROM python:3.10.6-buster


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY yolov8 yolov8
COPY setup.py setup.py
RUN pip install .


CMD uvicorn yolov8.api.fast:app --host 0.0.0.0
