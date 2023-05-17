from train import train

def main():
    split='DAY'
    use_tmp=False
    top_species=True
    n_classes=6 # (ELEPHANT/LION_FEMALE/DIKDIK/REEDBUCK/HIPPOPOTAMUS/EMPTY)
    output_file='checkpoint_day_n6'
    train(split, use_tmp, top_species, n_classes, output_file=output_file, viking=True)

if __name__ == '__main__':
    main()