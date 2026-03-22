
Mario Dataset - v6 2024-03-06 12:02pm
==============================

This dataset was exported via roboflow.com on March 20, 2026 at 1:31 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3117 images.
Mario-Objects are annotated in YOLO26 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Grayscale (CRT phosphor)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random rotation of between -14 and +14 degrees
* Random shear of between -14° to +14° horizontally and -15° to +15° vertically
* Random Gaussian blur of between 0 and 1.1 pixels
* Salt and pepper noise was applied to 0.46 percent of pixels


