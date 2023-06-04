This is an animal detection project based off of **[PyTorch](https://pytorch.org) Tutorial to Object Detection**.

The dataset used for training and validation is accessible on https://lila.science/datasets/snapshot-serengeti. The specific images used are listed in the bbox_images_split.csv file in the snapshot-serengeti folder.

The conda environment used to run the program can be setup using the environment.yaml file, using the command `conda env create -f environment.yaml`, and activated with `conda activate environment`.

### Executing Sample Code
Evaluation of a fully trained model (on a small sample of the validation dataset) can be performed by:
1. Activating the conda virtual environment on a machine with CUDA enabled
1. Change into the repository source directory
1. Run the command `python eval.py`.

To run object detection on a single image:
1. Activate the conda virtual environment 
1. Change into the repository source directory
1. Run the command `python detect.py`.
1. The image will be output to sample.png for inspection.

### Project Structures
#### Datasets
`datasets.py`: contains the class used to retrieve and handle the bounding box dataset
`snapshot-serengeti/`: contains dataset metadata and a sample of the images in the dataset

#### Transformations
`transformations.py`: code with the custom image and bounding box transformations, as well as train/validation sequential transforms

#### Utils
`utils.py`: miscelaneous utility functions

#### Eval
`eval.py`: functions for evaluating the mAP of various models.

#### Train
`train.py`: the main training code for the model

#### Models
`models/EfficientNetSSD300.py`: the SSD300 model with EfficientNetB2 base feature extractor. This was the final model used for training.
`models/EfficientNetSSD300.py`: the backup SSD300 model using MobileNet as the base feature extractor.