from torch.utils import data
from PIL import Image
import random
from collections import Counter


class BaseDataset(data.Dataset):
    """Base dataset class"""
    _GLOBAL_SEED = 12345

    def __init__(self, args, datalist, transforms, evaluate=False):
        self.args = args
        self.datalist = datalist
        self.transforms = transforms
        self.evaluate = evaluate

        # set global random seed to balance classes across processes
        self.global_random = random.Random()
        self.global_random.seed(self._GLOBAL_SEED)

        if self.args.balance_classes and not self.evaluate:
            self.balance_classes()

    def __len__(self):
        return len(self.datalist)

    def balance_classes(self):

        # get index of each class
        class_indices = {}
        for i, data_dict in enumerate(self.datalist):
            label = data_dict['label']
            if label not in class_indices:
                class_indices[label] = [i]
            else:
                class_indices[label].append(i)

        # get the max number of samples in a class
        max_len = max([len(class_indices[label]) for label in class_indices])

        # shuffle inds and balance classes
        balanced_indices = []
        for label in class_indices:
            self.global_random.shuffle(class_indices[label])
            q, r = divmod(max_len, len(class_indices[label]))
            balanced_indices.extend(class_indices[label] * q + class_indices[label][:r])

        # shuffle again to mix classes
        self.global_random.shuffle(balanced_indices)

        # update datalist
        self.datalist = [self.datalist[i] for i in balanced_indices]

        # print new stats
        if self.args.rank == 0:
            print(f'Balanced classes, new length: {len(self.datalist)}')
            prebalance_counter = {lab: len(dat) for lab, dat in class_indices.items()}
            print(f'Class counter before balancing: {prebalance_counter}')
            print(f'Class counter after balancing: {dict(Counter([item["label"] for item in self.datalist]))}')

    def transform_data(self, x):
        return self.transforms(x).float()

    def __getitem__(self, index):
        raise NotImplementedError


class SupervisedDataset(BaseDataset):
    """Dataset for supervised patch learning"""

    def __getitem__(self, index):
        image = Image.open(self.datalist[index]['image'].replace('/home/tonyxu74/scratch/patch_datasets/nct-crc-he-100k/', '/aippmdata/public/NCT-CRC-HE-100K/'))
        image = self.transform_data(image)
        label = self.datalist[index]['label']

        return image, label


class SimCLRDataset(BaseDataset):
    """Dataset for SSL with SimCLR"""

    def __init__(self, args, datalist, transforms, evaluate=False):
        super().__init__(args, datalist, transforms, evaluate=evaluate)
        # expand transform list
        self.transforms, self.multicrop_transforms = self.transforms

    def multicrop_transform_data(self, x):
        return self.multicrop_transforms(x).float()

    def __getitem__(self, index):
        image = Image.open(self.datalist[index]['path'].replace('/home/txu/scratch/amgrp/', '/project/amgrp/txu/'))

        # get SimCLR transformed large crops
        image_1 = self.transform_data(image)
        image_2 = self.transform_data(image)

        # get multicrops if required
        if self.args.multicrop > 0 and self.multicrop_transforms is not None:
            multicrop_images = [self.multicrop_transform_data(image) for _ in range(self.args.multicrop)]
            return image_1, image_2, *multicrop_images

        return image_1, image_2
