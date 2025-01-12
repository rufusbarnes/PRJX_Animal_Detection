import time
import pandas as pd
from copy import copy
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms.v2 as v2
from torchvision import disable_beta_transforms_warning
from models.EfficientNetSSD300 import EfficientNetSSD300, MultiBoxLoss
from datasets import SerengetiDataset, get_dataset_params
from transformations import *
from utils import *
import multiprocessing

def train(image_type, use_tmp, n_classes, output_file, checkpoint):
    """
    Training.
    :param image_type: 'night', 'day', or None
    :param use_tmp: bool, true to use tmp storage
    :param n_classes: int, number of classes including blank
    :param output_file: string, filename to save model to
    :param viking: bool, true if running on viking, false if running on gviking
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Learning parameters
    batch_size = 32  # batch size
    
    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    workers = num_cores
    print_freq = 50  # print training status every __ batches
    lr = 1e-3  # learning rate
    decay_lr_at = [80_000, 100_000]  # decay learning rate after these many iterations
    decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
    momentum = 0.9  # momentum
    weight_decay = 5e-4  # weight decay
    grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

    cudnn.benchmark = True

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = EfficientNetSSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.anchor_boxes).to(device)
    print(f'\nLoaded model to {device}, {workers} workers.')

    # Custom dataloaders
    params = get_dataset_params(use_tmp=use_tmp)
    train_dataset = SerengetiDataset(*params, split='train', image_type=image_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    print(f'\nInitialized data loader - use_tmp: {use_tmp}')

    epochs = 75
    # SET DECAY LR AT TO EPOCHS INSTEAD OF ITERATIONS
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    # Epochs
    print(f'\nTrain start.')
    for epoch in range(start_epoch, epochs+1):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train_epoch(train_loader=train_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch=epoch,
                    print_freq=print_freq,
                    grad_clip=grad_clip)

        # Save checkpoint
        # save_checkpoint(epoch, model, optimizer) REPLACE THIS
        if epoch % 25 == 0:
            print('Saving checkpoing:',output_file, epoch)
            save_checkpoint(epoch, model, optimizer, output_file=output_file+str(epoch))
        else:
            save_checkpoint(epoch, model, optimizer, output_file=output_file)

def train_epoch(train_loader, model, criterion, optimizer, epoch, print_freq, grad_clip):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

def main():
    image_type=None
    use_tmp=False
    n_classes=6 # (ELEPHANT/LION_FEMALE/DIKDIK/REEDBUCK/HIPPOPOTAMUS/EMPTY)
    output_file='checkpoint_full_n6'
    checkpoint='checkpoint_full_n6_epoch_40.pth.tar'
    train(image_type, use_tmp, n_classes, output_file=output_file, checkpoint=checkpoint)

if __name__ == '__main__':
    main()