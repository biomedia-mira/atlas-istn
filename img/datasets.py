import torch
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torchio as tio
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from img.processing import one_hot_labelmap


class ImageSegmentationOneHotDataset(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, csv_file_img, csv_file_seg, csv_file_msk=None, normalizer=None, resampler_img=None, resampler_seg=None, binarize=False, augmentation = False):
        """
        Args:
        :param csv_file_img (string): Path to csv file with image filenames.
        :param csv_file_seg (string): Path to csv file with segmentation filenames.
        :param csv_file_msk (string): Path to csv file with mask filenames.
        :param normalizer_img (callable, optional): Optional transform to be applied on each image.
        :param resampler_img (callable, optional): Optional transform to be applied on each image.
        :param normalizer_seg (callable, optional): Optional transform to be applied on each segmentation.
        :param resampler_seg (callable, optional): Optional transform to be applied on each segmentation.
        """
        self.img_data = pd.read_csv(csv_file_img)
        self.seg_data = pd.read_csv(csv_file_seg)

        if csv_file_msk:
            self.msk_data = pd.read_csv(csv_file_msk)

        self.augmentation = augmentation

        self.augment = tio.Compose([
            # tio.RandomAffine(p=0.5, scales = 0.05, degrees=5, default_pad_value=0)
            tio.RandomElasticDeformation(p=0.5, num_control_points=7, max_displacement=4, locked_borders=2)
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.img_data)), desc='Loading Data')):
            img_path = self.img_data.iloc[idx, 0]
            seg_path = self.seg_data.iloc[idx, 0]

            img_fname = os.path.basename(img_path)
            image = sitk.ReadImage(img_path, sitk.sitkFloat32)

            labelmap = sitk.ReadImage(seg_path, sitk.sitkInt64)

            if binarize == True:
                labelmap = sitk.Cast(labelmap > 0, sitk.sitkInt64)

            mask = sitk.GetImageFromArray(np.ones(image.GetSize()[::-1]))
            mask.CopyInformation(image)

            if csv_file_msk:
                msk_path = self.msk_data.iloc[idx, 0]

                mask = sitk.ReadImage(msk_path, sitk.sitkUInt8)

            if normalizer:
                image = normalizer(image, mask)

            if resampler_img:
                image = resampler_img(image)

            if resampler_seg:
                labelmap = resampler_seg(labelmap)
                mask = resampler_seg(mask)

            if len(image.GetSize()) == 3:
                image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
                labelmap.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
                mask.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            else:
                image.SetDirection((1, 0, 0, 1))
                labelmap.SetDirection((1, 0, 0, 1))
                mask.SetDirection((1, 0, 0, 1))

            image.SetOrigin(np.zeros(len(image.GetOrigin())))
            labelmap.SetOrigin(np.zeros(len(image.GetOrigin())))
            mask.SetOrigin(np.zeros(len(image.GetOrigin())))

            labelmap = sitk.Cast(one_hot_labelmap(labelmap, smoothing_sigma=0.5), sitk.sitkVectorFloat32)

            sample = {'image': image, 'labelmap': labelmap, 'mask': mask, 'fname': img_fname}

            self.samples.append(sample)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item):
        sample = self.samples[item]

        image = torch.from_numpy(sitk.GetArrayFromImage(sample['image'])).unsqueeze(0)
        mask = torch.from_numpy(sitk.GetArrayFromImage(sample['mask'])).unsqueeze(0)

        if len(image.size()) == 4:
            labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(3, 0, 1, 2)
        else:
            labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(2, 0, 1)

        if self.augmentation:
            subject_dict = {
                'image': tio.ScalarImage(tensor=image),
                'mask': tio.LabelMap(tensor=mask),
                'labelmap': tio.ScalarImage(tensor=labelmap),
            }
            subject = self.augment(tio.Subject(subject_dict))
            return {'image': subject['image'].data, 'labelmap': subject['labelmap'].data, 'mask': subject['mask'].data}
        else:
            return {'image': image, 'labelmap': labelmap, 'mask': mask}

    def get_sample(self, item):
        return self.samples[item]


class ImageSegmentationOneHotDatasetFromDisk(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, csv_file_img, csv_file_seg, csv_file_msk=None, normalizer=None, resampler_img=None, resampler_seg=None, binarize=False, augmentation = False):
        """
        Args:
        :param csv_file_img (string): Path to csv file with image filenames.
        :param csv_file_seg (string): Path to csv file with segmentation filenames.
        :param csv_file_msk (string): Path to csv file with mask filenames.
        :param normalizer_img (callable, optional): Optional transform to be applied on each image.
        :param resampler_img (callable, optional): Optional transform to be applied on each image.
        :param normalizer_seg (callable, optional): Optional transform to be applied on each segmentation.
        :param resampler_seg (callable, optional): Optional transform to be applied on each segmentation.
        """
        self.img_data = pd.read_csv(csv_file_img)
        self.seg_data = pd.read_csv(csv_file_seg)

        if csv_file_msk:
            self.msk_data = pd.read_csv(csv_file_msk)

        self.normalizer = normalizer
        self.resampler_img = resampler_img
        self.resampler_seg = resampler_seg
        self.binarize = binarize
        self.augmentation = augmentation

        self.augment = tio.Compose([
            # tio.RandomAffine(p=0.5, scales = 0.05, degrees=5, default_pad_value=0)
            tio.RandomElasticDeformation(p=0.5, num_control_points=7, max_displacement=4, locked_borders=2)
        ])

        self.sample_paths = []
        for idx in range(len(self.img_data)):
            img_path = self.img_data.iloc[idx, 0]
            seg_path = self.seg_data.iloc[idx, 0]

            if csv_file_msk:
                msk_path = self.msk_data.iloc[idx, 0]
            else:
                msk_path = None

            sample_path = {'img_path': img_path, 'seg_path': seg_path, 'msk_path': msk_path}
            self.sample_paths.append(sample_path)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sitk.GetArrayFromImage(sample['image'])).unsqueeze(0)
        mask = torch.from_numpy(sitk.GetArrayFromImage(sample['mask'])).unsqueeze(0)

        if len(image.size()) == 4:
            labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(3, 0, 1, 2)
        else:
            labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(2, 0, 1)

        if self.augmentation:
            subject_dict = {
                'image': tio.ScalarImage(tensor=image),
                'mask': tio.LabelMap(tensor=mask),
                'labelmap': tio.ScalarImage(tensor=labelmap),
            }
            subject = self.augment(tio.Subject(subject_dict))
            return {'image': subject['image'].data, 'labelmap': subject['labelmap'].data, 'mask': subject['mask'].data}
        else:
            return {'image': image, 'labelmap': labelmap, 'mask': mask}

    def get_sample(self, item):
        sample_path = self.sample_paths[item]

        img_fname = os.path.basename(sample_path['img_path'])
        image = sitk.ReadImage(sample_path['img_path'], sitk.sitkFloat32)

        labelmap = sitk.ReadImage(sample_path['seg_path'], sitk.sitkInt64)

        if self.binarize == True:
            labelmap = sitk.Cast(labelmap > 0, sitk.sitkInt64)

        mask = sitk.GetImageFromArray(np.ones(image.GetSize()[::-1]))
        mask.CopyInformation(image)

        if sample_path['msk_path']:
            mask = sitk.ReadImage(sample_path['msk_path'], sitk.sitkUInt8)

        if self.normalizer:
            image = self.normalizer(image, mask)

        if self.resampler_img:
            image = self.resampler_img(image)

        if self.resampler_seg:
            labelmap = self.resampler_seg(labelmap)
            mask = self.resampler_seg(mask)

        if len(image.GetSize()) == 3:
            image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            labelmap.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            mask.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        else:
            image.SetDirection((1, 0, 0, 1))
            labelmap.SetDirection((1, 0, 0, 1))
            mask.SetDirection((1, 0, 0, 1))

        image.SetOrigin(np.zeros(len(image.GetOrigin())))
        labelmap.SetOrigin(np.zeros(len(image.GetOrigin())))
        mask.SetOrigin(np.zeros(len(image.GetOrigin())))

        labelmap = sitk.Cast(one_hot_labelmap(labelmap, smoothing_sigma=0.5), sitk.sitkVectorFloat32)

        sample = {'image': image, 'labelmap': labelmap, 'mask': mask, 'fname': img_fname}

        return sample
