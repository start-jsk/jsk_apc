from .base import class_names_apc2016  # NOQA
from .jsk import JskAPC2016Dataset  # NOQA
from .mit import MitAPC2016Dataset  # NOQA


if __name__ == '__main__':
    from ...utils import view_class_seg_dataset
    import argparse

    datasets = [
        'JskAPC2016Dataset',
        'MitAPC2016Dataset',
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=datasets, required=True)
    parser.add_argument('--split', default='valid')
    parser.add_argument('--locations', nargs='+', default=['shelf'])
    args = parser.parse_args()

    if args.dataset == 'JskAPC2016Dataset':
        dataset = JskAPC2016Dataset(args.split)
    elif args.dataset == 'MitAPC2016Dataset':
        dataset = MitAPC2016Dataset(args.split, locations=args.locations)
    view_class_seg_dataset(dataset)
