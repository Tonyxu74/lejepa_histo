from torch.utils import data
from PIL import Image
import torch
import dplabtools.config.slides
dplabtools.config.slides.mpp_round_decimal_places = 4
from dplabtools.slides.patches import WholeImageGridPatches, MemPatchExtractor
#from dplabtools.slides.processing import WSIMask
import os
import numpy as np
import random
import h5py
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


class FeatureExtractionDataset(BaseDataset):
    """Dataset for extracting feature bags out of entire WSIs"""

    def __init__(self, args, test_datalist, transforms, return_dims=False):
        super().__init__(args, test_datalist, transforms, evaluate=True)
        self.return_dims = return_dims
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.datalist) - 1:
            self.index = -1
            raise StopIteration
        self.index += 1
        return self.__getitem__(self.index)

    def wsi_generator(self, patch_generator, patch_data):

        # patch generator will extract patches out of order, pass thru model in batches
        patch_batch = []
        patch_locs = []
        for patch_num, (patch, label, patch_index) in enumerate(patch_generator):
            # yield the batch, reset batch list
            if patch_num != 0 and patch_num % self.args.batch_size == 0:
                yield torch.stack(patch_batch), patch_locs
                patch_batch = []
                patch_locs = []

            # get a transformed patch and its location
            patch_batch.append(self.transform_data(patch))
            patch_locs.append(patch_data[patch_index][0])

        # yield the last batch too
        yield torch.stack(patch_batch), patch_locs

    def __getitem__(self, index):
        # extract patches from WSI at index
        # pass to wsi_generator
        wsi_path = self.datalist[index]['path']
        wsi_label = self.datalist[index]['label']
        patient_id = self.datalist[index]['patient']

        if '/CAMELYON16/' in wsi_path:
            orig_dataset = 'CAMELYON16'
        elif '/CAMELYON17/' in wsi_path:
            orig_dataset = 'CAMELYON17'
        elif '/TBLN/' in wsi_path:
            orig_dataset = 'TBLN'
            if 'testing_rescanned' in wsi_path:
                orig_dataset = 'TBLN_rescanned'
        elif '/MSK_Lymph_Node_data/' in wsi_path:
            orig_dataset = 'MSK'
        else:
            raise ValueError(f'Unknown dataset: {wsi_path}')

        wsi_id = wsi_path.split("/")[-1].split(".")[0]

        mask_path = f'{self.args.mask_base_path}/{orig_dataset}/{wsi_id}.npy'
        if os.path.exists(mask_path):
            tissue_mask = np.load(mask_path)
        else:
            try:
                tissue_mask = WSIMask(wsi_file=wsi_path, level=2)
            except:
                print('Warning: level 2 not available! Using level 1 for mask creation.')
                tissue_mask = WSIMask(wsi_file=wsi_path, level=1)
            tissue_mask.save_array(mask_path)
            tissue_mask = tissue_mask.array

        patches = WholeImageGridPatches(
            wsi_file=wsi_path,
            mask_data=tissue_mask,
            patch_size=224,
            foreground_threshold=0.5,
            level_or_mpp=0.5,
            patch_stride=1
        )
        extractor = MemPatchExtractor(
            patches=patches,
            num_workers=8,
            inference_mode=True,
            resampling_mode='tile'
        )
        if self.return_dims:
            slide_dims = patches._wsi_slide.level_dimensions[0]
            try:
                slide_level2_dims = patches._wsi_slide.level_dimensions[2]
            except:
                print('Warning: level 2 not available! Logging level 1 dimensions.')
                slide_level2_dims = patches._wsi_slide.level_dimensions[1]
                
            mpp_x, mpp_y = patches._wsi_slide.mpp_data

            return self.wsi_generator(extractor.patch_stream, patches.patch_data), wsi_id, slide_dims, \
                slide_level2_dims, (mpp_x, mpp_y), wsi_label, patient_id

        return self.wsi_generator(extractor.patch_stream, patches.patch_data), wsi_id, wsi_label, patient_id


class MILDataset(BaseDataset):
    """Dataset for doing MIL with weak slide labels"""

    def __init__(self, args, datalist, transforms, evaluate=False):
        super().__init__(args, datalist, transforms, evaluate=evaluate)
        self.fold_features = np.array([int(x) for x in args.fold_features.split(',')])

    def __getitem__(self, index):
        wsi_features = h5py.File(self.datalist[index]['path'], 'r')

        # get all features, merge feats from each fold to feature dimension
        all_features = wsi_features['all_features'][self.fold_features, :, :]
        # all_features = self.transform_data(all_features).permute(1, 0, 2).flatten(start_dim=1)
        all_features = list(self.transform_data(all_features))

        # get the label
        if self.datalist[index]['label'] == 'positive':
            label = 1
        elif self.datalist[index]['label'] == 'negative':
            label = 0
        else:
            raise ValueError(f'Unknown label: {self.datalist[index]["label"]}')

        # cat along channel dim
        return torch.cat(all_features, dim=1), label


class SimCLRDataset(BaseDataset):
    """Dataset for PAWS unlabelled images"""

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


class PAWSLabelledDataset:
    """Dataset for PAWS labelled support samples"""

    _INIT_GLOBAL_SEED = 12345

    def __init__(self, args, datalist, transforms, smoothing=0.1):

        # get a global random number generator to make sure each process gets a mutually exclusive subset of data
        self.global_random = random.Random()
        self.global_seed = self._INIT_GLOBAL_SEED
        # get a local random number generator to grab batches, seed should be different for each rank
        self.local_random = random.Random()
        self.local_random.seed(self.args.rank)

        self.args = args
        self.datalist = datalist
        self.transforms = transforms

        self.class_dict = self.get_class_dict()

        # get number of classes in dataset
        self.num_classes = len(self.class_dict.keys())
        self.smoothing = smoothing
        self.labels_matrix = self.create_labels_matrix()

        # 0. split datalist on class of inputs
        # 1. shuffle, then split integers based on world_size and rank (equally in each class)
        # 2. implement a "labels_matrix" code and save to self.labels_matrix so it can be reused (depends on world size
        #  and batch size
        # 3. when getting a batch, sample batch_size amount (without replacement) per class from the proper indices,
        #  and return them concatenated along the batch dimension (make sure the format is [a,b,c,a,b,c,a,b,c...])

    def transform_data(self, x):
        return self.transforms(x).float()

    def get_class_dict(self):
        class_dict = {}

        # separate classes using class_dict
        for datapt in self.datalist:
            label = datapt['label']
            if label in class_dict:
                class_dict[label].append(datapt)
            else:
                class_dict[label] = [datapt]
        return class_dict

    def get_rank_class_dict(self):
        self.global_random.seed(self.global_seed)
        rank_class_dict = {}

        # go through classes separately, split data for each process
        for label in self.class_dict:
            # number of items in class
            items_in_class = len(self.class_dict[label])
            data_inds = [i for i in range(items_in_class)]
            self.global_random.shuffle(data_inds)

            print('Delete this later, data_inds = ', data_inds[:30], 'rank = ', self.args.rank)

            # number of items in rank for distributed training
            items_in_rank = items_in_class // self.args.world_size
            rank_data_inds = data_inds[self.args.rank*items_in_rank: (self.args.rank+1)*items_in_rank]

            print('Delete this later, rank_data_inds = ', rank_data_inds[:30], 'rank = ', self.args.rank)

            # populate new dict with subsampled data_inds
            rank_class_dict[label] = rank_data_inds

            # check we have enough support images per class
            assert len(rank_class_dict[label]) > self.args.support_batch_size, \
                'Number of images in a class must be greater than support batch size!'

        return rank_class_dict

    def create_labels_matrix(self):
        # total images = local_images * world_size since we gather the support representations from all processes
        local_images = self.args.support_batch_size * self.num_classes
        total_images = local_images * self.args.world_size

        # offset value for label smoothing
        offset_value = self.smoothing/self.num_classes
        labels_matrix = torch.zeros(total_images, self.num_classes) + offset_value

        # get positive class
        for i in range(self.num_classes):
            labels_matrix[i::self.num_classes][:, i] = 1. - self.smoothing + offset_value
        return labels_matrix

    def next_epoch(self):
        # iterate global_seed by 1 to generate a new split for the next epoch
        self.global_seed += 1

    def get_support_images(self):
        # get a dict of indices for self.class_dict that is distributed among each process with no overlap
        rank_class_dict = self.get_rank_class_dict()

        support_batch = [None for _ in range(self.args.support_batch_size * self.num_classes)]

        for class_num, label in enumerate(rank_class_dict):

            # random sample with self.local_random so each process gets indices differently
            sampled_inds = self.local_random.sample(rank_class_dict[label], k=self.args.support_batch_size)
            print('Delete this later, sampled_inds = ', sampled_inds[:30], 'rank = ', self.args.rank)

            # load support images and perform transforms
            sampled_data_paths = [self.class_dict[label][i]['path'] for i in sampled_inds]
            sampled_data = [self.transform_data(Image.open(path)) for path in sampled_data_paths]

            # return classes in same order as self.labels_matrix
            support_batch[class_num::self.num_classes] = sampled_data

        return torch.stack(support_batch), self.labels_matrix


