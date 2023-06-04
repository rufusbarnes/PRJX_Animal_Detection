This is an animal detection project based off of **[PyTorch](https://pytorch.org) Tutorial to Object Detection**.

The dataset used for training and validation is accessible on https://lila.science/datasets/snapshot-serengeti. The specific images used are listed in the bbox_images_split.csv file in the snapshot-serengeti folder.

The conda environment used to run the program can be setup using the environment.yaml file, using the command `conda env create -f environment.yaml`, and activated with `conda activate environment`.

Evaluation of a fully trained model (on a small sample of the validation dataset) can be performed by:
1. Activating the conda virtual environment on a machine with CUDA enabled
1. Change into the repository source directory
1. Run the command `python eval.py`.