
chess pieces 2 - v3 2023-12-01 6:05pm
==============================

This dataset was exported via roboflow.com on February 15, 2024 at 4:32 PM GMT

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

The dataset includes 15622 images.
Chess-pieces-VRka are annotated in YOLOv8 Oriented Object Detection format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random brigthness adjustment of between -15 and +15 percent
* Salt and pepper noise was applied to 3 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Salt and pepper noise was applied to 3 percent of pixels


