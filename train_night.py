from train import train

def main():
    split='NIGHT'
    use_tmp=False
    n_classes=6 # (ELEPHANT/LION_FEMALE/DIKDIK/REEDBUCK/HIPPOPOTAMUS/EMPTY)
    output_file='checkpoint_night_n6'
    train(split, use_tmp, n_classes, output_file=output_file, viking=True)

if __name__ == '__main__':
    main()