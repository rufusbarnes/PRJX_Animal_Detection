from utils import *
from datasets import SerengetiDataset, get_dataset_params
from tqdm import tqdm
from pprint import PrettyPrinter
import time
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

def evaluate(test_loader, model, min_score=0.1, max_overlap=0.3, top_k=6):
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
        for _, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='EVALUATING')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            '''
            THIS IS WHERE YOU CAN CHANGE MIN SCORE, MAX OVERLAP, TOP_K
            '''
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=min_score,
                                                                                       max_overlap=max_overlap,
                                                                                       top_k=top_k)
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
        APs, mAP, precision, recall = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)

    print('\nMean Average Precision (mAP): %.3f' % mAP)
    return APs, mAP, precision, recall


def eval_best(batch_size, workers, min_score, max_overlap, top_k, test_datasets):
    best_checkpoints = ['full_150', 'full_100', 'day_100', 'night_100']
    get_path = lambda x: './models/' + x + '.pth.tar'
    checkpoints = [get_path(checkpoint) for checkpoint in best_checkpoints]

    print(f'\nUsing device: {device}.')
    for filename in checkpoints:
        print(f'\nEvaluating checkpoint: {filename}.')    
        checkpoint = torch.load(filename)
        model = checkpoint['model']
        model = model.to(device)
        model.eval()
        for image_type, test_dataset in test_datasets.items():
            print(f'\nUsing {image_type} images.')
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                    collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
            print(evaluate(test_loader, model, min_score, max_overlap, top_k))

def eval_all(batch_size, workers, min_score, max_overlap, top_k, test_datasets):
    mixed_checkpoints  = ['full_100', 'full_150']
    day_checkpoints   = ['day_100'] 
    night_checkpoints = ['night_100']
    checkpoints = {
        'mixed': mixed_checkpoints,
        'day': day_checkpoints,
        'night': night_checkpoints
    }

    get_path = lambda x: './models/' + x + '.pth.tar'
    for k, v in checkpoints.items():
        test_dataset = test_datasets[k]
        for f in v:
            path = get_path(f)
            checkpoint = torch.load(path)
            model = checkpoint['model']
            model = model.to(device)
            model.eval()
            print(f'Testing model {f}')
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                    collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
            print(evaluate(test_loader, model, min_score, max_overlap, top_k))

def eval_single(batch_size, workers, min_score, max_overlap, top_k, test_dataset):
    f = 'full_150'
    get_path = lambda x: '../models/' + x + '.pth.tar'
    path = get_path(f)

    checkpoint = torch.load(path)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    print(f'Testing model {f}')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
    return evaluate(test_loader, model, min_score, max_overlap, top_k)

def eval_cross_val(batch_size, workers, min_score, max_overlap, top_k, test_dataset):
    mAPs = []
    min_scores = [0.01] + [0.05*i for i in range(1,11)]
    max_overlaps = [0.05*i for i in range(1,11)]
    for min_score in min_scores:
        for max_overlap in max_overlaps:
            print(f'Min Score: {min_score}')
            print(f'Max Overlap: {max_overlap}')
            print(f'Top K: {top_k}')

            APs, mAP = eval_single(batch_size, workers, min_score, max_overlap, top_k, test_dataset)
                # Print AP for each class
            obj = {
                'min_score':min_score,
                'max_overlap':max_overlap,
                'top_k': top_k,
                'mAP':mAP,
                'APs':APs
            }
            mAPs.append(obj)
            pp.pprint(APs)
            print(f'mAP: {mAP}.')
    return mAPs

def main():
    '''
    Script to evaluate multiple models on mixed, day and night datasets
    '''
    # Parameters
    num_cores = multiprocessing.cpu_count()
    workers = num_cores
    print('Workers:',workers)
    batch_size = 2 * workers

    mixed = SerengetiDataset(*get_dataset_params(use_tmp=False), split='test')
    day = SerengetiDataset(*get_dataset_params(use_tmp=False), split='test', image_type='day')
    night = SerengetiDataset(*get_dataset_params(use_tmp=False), split='test', image_type='night')
    test_datasets = {'mixed': mixed, 'day': day, 'night': night}

    top_k=6
    min_score = 0.2   # anything 0.05 - 0.2 works well, the higher the value the faster the training
    max_overlap = 0.4 # 0.3 - 0.4 works best on full_75
    
    eval_best(batch_size, workers, min_score, max_overlap, top_k, test_datasets)
    eval_all(batch_size, workers, min_score, max_overlap, top_k, test_datasets)

if __name__ == '__main__':
    main()
