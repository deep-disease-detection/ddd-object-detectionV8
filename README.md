# ddd-object-detectionV8

We used the [YOLOv8 from ultralytics](https://github.com/ultralytics/ultralytics) package that was very easy to initialize, train and deploy.

The most important part of using YOLOv8 for virus detection was preprocessing the data. The initial annotations did not include bounding boxes and they were difficult to infer for certain types of viruses.

## Set-up
Make sure you install our other package, [ddd-ai](https://github.com/deep-disease-detection/ddd-ai), which is re-used in the data preprocessing steps.


## The preprocessing
The most important aspect of training an object detection algorithm to detect viruses on microscopes was generating the appropriate training data.
The annotations in the raw data only included the center of the virus particles or several points along the central line of elongated viruses. From this data, we needed to generate bounding boxes appropriate for training a YOLO model.

### Basic preprocessing steps
- **Loading and formatting** the data from the image folders and the annotations : virus particle positions and class
- **Resize** the image: Microscope images have different resolutions (nm per pixel). Using the meta-data of the picture, we make sure to rescale the X and Y axis of the image to have the same resolution accross all images. We used LANCZOS3 Kernel interpolation to resize the images. Particle positions are also rescaled accordingly.
- **Crop the image**: We compute the center of all virus particles on the image and crop around it. We add padding (zero-padding) when required. If padding was added, we also adapt the position of the particles.

### Computing bounding boxes
A dictionnary in the params.py file of the package contains meta-data about each virus class, including whether it is elongated and its usual diamater in nm.
- For circular viruses, we simply center the bounding box around the particle center provided in the annotation file. The width and height of the box are the diameter of the virus.
- For elongated viruses, we compute the min and max of the x and y coordinates of all the points on the center line of the virus. We add a little bit of margin to ensure the virus is fully included in the bounding box.

### Augmentation
🚧 Work in Progress 🚧

## The YOLOv8 model
The model was trained on google colab using the YOLOv8 API.

After 100 epochs, we reached 0,9 mAP on our validation dataset.
