# ddd-object-detectionV8

We used the [YOLOv8 from ultralytics](https://github.com/ultralytics/ultralytics) package that was very easy to initialize, train and deploy.

The most important part of using YOLOv8 for virus detection was preprocessing the data. The initial annotations did not include bounding boxes and they were difficult to infer for certain types of viruses.

## The preprocessing
🚧 Work in Progress 🚧

## The YOLOv8 model
The model was trained on google colab using the YOLOv8 API.

After 100 epochs, we reached 0,9 mAP.
