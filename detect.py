from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from datasets import show_sample
from datasets import *
import os, random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    det_scores = [score for score in det_scores[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    font_size = 60
    font = ImageFont.truetype("Arial.ttf", font_size)
    
    draw = ImageDraw.Draw(annotated_image)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper() + ': 0.000')
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=f'{det_labels[i].upper()}: {det_scores[i]:.3f}', fill='white', font=font)
    del draw
    return annotated_image


if __name__ == '__main__':
    # Select a random image
    image_folder = '../snapshot-serengeti/'
    test_images = pd.read_csv('./snapshot-serengeti/bbox_images_split.csv')
    test_images = test_images[test_images['split'] == 'test'] 
    test_images = SerengetiDataset(*get_dataset_params(), split='TEST').images_df
    img = random.choice(test_images['image_path_rel'].values)

    print(f'\nLoaded image {img}.')
    img = os.path.join(image_folder, img)
    img = Image.open(os.path.join(image_folder, img))

    # Load model checkpoint
    checkpoint = '../full_75.pth.tar'
    checkpoint = torch.load(checkpoint, map_location=device)
    print(f'\nLoaded checkpoint from epoch {checkpoint["epoch"] + 1}.\n')
    model = checkpoint['model'].to(device)
    model = model.to(device)

    #Save image to sample.png
    filename = 'sample.png'
    detect(img, min_score=0.1, max_overlap=0.3, top_k=6, suppress={'elephant'}).save(filename)
    print(f'\nSaved detected image to {filename}.')