import torch
from torch.utils.data import Dataset
from transformations import BBoxToBoundary, BBoxSerengetiToBoundary, train_transform, test_transform
import os
import pandas as pd
from PIL import Image
from ast import literal_eval
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

class SerengetiDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, image_folder, images_df, annotations_df, classes_df, night_images, split, image_type=None):
        with open('./snapshot-serengeti/annotations_map.json', 'r') as f:
            self.bboxes = json.load(f)
        
        self.image_folder = image_folder
        self.annotations_df = annotations_df
        self.images_df = images_df
        
        self.split = split.lower()
        assert self.split in {'train', 'test'}
        self.images_df = self.images_df[self.images_df['split'] == self.split]
        if self.split == 'train':
            self.transform = train_transform()
        else:
            self.transform = test_transform()
        
        self.classes_df = classes_df
        present_classes = {name.lower() for name in set(self.images_df['question__species'].values)}  # Get set of present species
        present_classes.add('blank')
        self.classes_df = self.classes_df[self.classes_df['name'].isin(present_classes)] # Filter out missing animals
        self.classes_df['id'] = self.classes_df.reset_index().index                      # reset ids to 0 to n - 1

        self.night_images = night_images
        self.image_type = image_type
        if self.image_type:
            self.image_type = self.image_type.upper()
            assert self.image_type in {'DAY', 'NIGHT'}
            if self.image_type == 'NIGHT':
                self.images_df = self.images_df[self.images_df['image_path_rel'].isin(self.night_images)]
            elif self.image_type == 'DAY':
                self.images_df = self.images_df[~self.images_df['image_path_rel'].isin(self.night_images)]

        self.annotations_df['bbox'] = self.annotations_df['bbox'].apply(literal_eval)
        
        print(f'\nInitialized dataset - split: {self.split} / classes: {len(self.classes_df)}.')

    def __getitem__(self, i):
        image_info = self.images_df.iloc[i]

        path = os.path.join(self.image_folder, image_info['image_path_rel'])
        image = Image.open(path)

        box_idxs = self.bboxes[image_info['id']]
        boxes = torch.FloatTensor([self.annotations_df.iloc[i]['bbox'] for i in box_idxs])

        species = image_info['question__species'].lower()
        label = self.classes_df.loc[self.classes_df['name'] == species, 'id'].iloc[0]
        labels = torch.FloatTensor([label for _ in boxes])
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        return image, boxes, labels

    def __len__(self):
        return len(self.images_df)

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, x, y), 3 lists of N tensors each

    def get_classes(self):
        classes_present = set()
        for _, row in self.images_df.iterrows():
            species = row['question__species'].lower()
            classes_present.add(species)
    
        filtered_classes = self.classes_df[self.classes_df['name'].isin(classes_present)]
        return (filtered_classes)

    def get_class_frequencies(self):
        class_frequencies = {row['name']: 0 for i, row in self.get_classes().iterrows()}
        for i, row in self.images_df.iterrows():
            species = row['question__species'].lower()
            box_idxs = self.bboxes[row['id']]
            class_frequencies[species] += len(box_idxs)
        
        return class_frequencies

def show_sample(sample, bbox_type=None):
    if bbox_type == 'fractional':
        sample = BBoxToBoundary()(sample)
    elif bbox_type == 'serengeti':
        sample = BBoxSerengetiToBoundary()
    image, bboxes, labels = sample

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.title(labels)
    for bbox in bboxes:
        #bottom left, width, height
        x = bbox[0] 
        y = bbox[1]
        w = abs(bbox[2] - bbox[0])
        h = abs(bbox[3] - bbox[1])
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def get_dataset_params(use_tmp=False):
    '''
    Utility function holding parameters used to initialize a dataset
    :param use_tmp: set to true if image data is being stored in the GPU /tmp store
    '''
    if use_tmp:
        image_folder = '/tmp/snapshot-serengeti/'
    else:
        image_folder = './snapshot-serengeti/images/'

    images_df = pd.read_csv('./snapshot-serengeti/test_sample.csv')
    
    annotations_df = pd.read_csv('./snapshot-serengeti/bbox_annotations_downloaded.csv')
    classes_df = pd.read_csv('./snapshot-serengeti/classes.csv')
    with open('./snapshot-serengeti/grayscale_images.txt', 'r') as f:
        night_images = set(ast.literal_eval(f.read()))

    return image_folder, images_df, annotations_df, classes_df, night_images


def main():
    # dataset = SerengetiDataset(*get_dataset_params())
    day_dataset = SerengetiDataset(*get_dataset_params(), split='train', image_type='DAY')
    night_dataset = SerengetiDataset(*get_dataset_params(), split='train', image_type='NIGHT')
    day_freqs = day_dataset.get_class_frequencies()
    night_freqs = night_dataset.get_class_frequencies()
    total_freqs = {k: (v, night_freqs[k]) for k, v in day_freqs.items() if k in night_freqs.keys()}
    viable_freqs = {k: v for k, v in total_freqs if min(v) > 500}
    print(total_freqs)


if __name__ == '__main__':
    main()