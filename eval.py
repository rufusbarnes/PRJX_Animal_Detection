from utils import *
from datasets import SerengetiDataset, get_dataset_params
from tqdm import tqdm
from pprint import PrettyPrinter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    start = time.time()

    with torch.no_grad():
        # Batches
        for _, (images, boxes, labels) in enumerate((test_loader)):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            '''
            THIS IS WHERE YOU CAN CHANGE MIN SCORE, MAX OVERLAP, TOP_K
            '''
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.15,
                                                                                       max_overlap=0.45,
                                                                                       top_k=6)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)

            batch_time.update(time.time() - start)
            start = time.time()
            # print(f'Data time: {data_time.val}, Batch time: {batch_time.val}')

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


def eval_full():
    # Parameters
    batch_size = 64
    workers = 32
    checkpoint = '../full_75.pth.tar'
    print(f'\nEvaluating checkpoint: {checkpoint}.')

    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint)
    
    model = checkpoint['model']
    model = model.to(device)
    print(f'\nLoaded model to device {device}')
    print('\nUsing both day and night images.')

    # Switch to eval mode
    model.eval()

    # Load test data
    test_dataset = SerengetiDataset(*get_dataset_params(use_tmp=False), split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

    evaluate(test_loader, model)


def main():
    '''
    Script to evaluate multiple models on mixed, day and night datasets
    '''
    # Parameters
    batch_size = 2
    workers = 1
    checkpoints = ['../full_75.pth.tar']

    mixed = SerengetiDataset(*get_dataset_params(use_tmp=False), split='test')
    day = SerengetiDataset(*get_dataset_params(use_tmp=False), split='test', image_type='day')
    test_datasets = {'mixed':mixed,'day': day}
    # night = SerengetiDataset(*get_dataset_params(use_tmp=False), split='test', image_type='night')
    # test_datasets = {'mixed': mixed, 'day': day, 'night': night}

    print(f'\nUsing device: {device}.')
    for filename in checkpoints:
        print(f'\nEvaluating checkpoint: {filename}.')    
        checkpoint = torch.load(filename)
        model = checkpoint['model']
        model = model.to(device)
        # Switch to eval mode
        model.eval()
        for image_type, test_dataset in test_datasets.items():
            print(f'\nUsing {image_type} images.')
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                    collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
            evaluate(test_loader, model)

if __name__ == '__main__':
    main()
