from train import train

def main():
    split='DAY'
    use_tmp=False
    n_classes=6 # (ELEPHANT/LION_FEMALE/DIKDIK/REEDBUCK/HIPPOPOTAMUS/EMPTY)
    output_file='checkpoint_day_n6'
    checkpoint=None
    train(split, use_tmp, n_classes, output_file=output_file, checkpoint=checkpoint)

if __name__ == '__main__':
    main()