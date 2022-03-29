import os
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

import numpy as np
import matplotlib as mpl

back_end = mpl.get_backend()
try:
    mpl.use('module://backend_interagg')
    import matplotlib.pyplot as plt

    print('Set matplotlib backend to interagg')
except ImportError:
    print('Cannot set matplotlib backend to interagg, resorting to default backend {}'.format(back_end))
    mpl.use(back_end)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print('Cannot set matplotlib backend to interagg, resorting to default backend {}'.format(back_end))
    mpl.use(back_end)
    import matplotlib.pyplot as plt

import SimpleITK as sitk

from nets.convnet import UNet2D, UNet3D
from nets.stn import FullSTN2D, FullSTN3D, DiffeomorphicSTN2D, DiffeomorphicSTN3D, AffineSTN2D, AffineSTN3D
from img.processing import zero_mean_unit_var
from img.processing import range_matching
from img.processing import zero_one
from img.processing import threshold_zero
from img.transforms import Resampler
from img.transforms import Normalizer
from img.datasets import ImageSegmentationOneHotDataset
import utils.metrics as mira_metrics
import utils.tensorboard_helpers as mira_th
import utils.vis_helpers as mira_vis
from tensorboardX import SummaryWriter
from attrdict import AttrDict

separator = '----------------------------------------'

# torch.autograd.set_detect_anomaly(True)

def write_images(writer, phase, image_dict, n_iter, mode3d):
    for name, image in image_dict.items():
        if mode3d:
            if image.size(1) == 1:
                # writer.add_image('{}/{}'.format(phase, name), mira_th.volume_to_batch_image(image), n_iter)
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, :, int(image.size(2)/2), ...]), n_iter)
                # writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, :, :, int(image.size(3) / 2), :]), n_iter)
                # writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, :, :, :, int(image.size(4) / 2)]), n_iter)
            elif image.size(1) > 3:
                # writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, 1:4, int(image.size(2) / 2), ...]), n_iter, dataformats='CHW')
                writer.add_image('{}/{}'.format(phase, name),
                                 torch.clamp(image[0, 1:4, int(image.size(2) / 2), ...], 0, 1), n_iter,
                                 dataformats='CHW')
            else:
                writer.add_image('{}/{}'.format(phase, name),
                                 mira_th.normalize_to_0_1(image[0, 1, int(image.size(2) / 2), ...]), n_iter,
                                 dataformats='HW')
        else:
            if image.size(1) ==  1:
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, ...]), n_iter)
            elif image.size(1) > 3:
                # writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, 1:4, ...]), n_iter, dataformats='CHW')
                writer.add_image('{}/{}'.format(phase, name), torch.clamp(image[0, 1:4, ...], 0, 1), n_iter,
                                 dataformats='CHW')
            else:
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, 1, ...]), n_iter, dataformats='HW')


def write_values(writer, phase, value_dict, n_iter):
    for name, value in value_dict.items():
        writer.add_scalar('{}/{}'.format(phase, name), value, n_iter)


def set_up_model_and_preprocessing(phase, args):
    print(separator)
    print('Starting {}...'.format(phase))
    print(separator)

    with open(args.config) as f:
        config = json.load(f)

    print('Config from file: ' + str(config))

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")

    print('Device: ' + str(device))
    if use_cuda:
        print('GPU: ' + str(torch.cuda.get_device_name(int(args.dev))))

    if args.stn == 'f':
        if args.mode3d:
            stn_model = FullSTN3D
        else:
            stn_model = FullSTN2D
    elif args.stn == 's':
        if args.mode3d:
            stn_model = DiffeomorphicSTN3D
        else:
            stn_model = DiffeomorphicSTN2D
    elif args.stn == 'a':
        if args.mode3d:
            stn_model = AffineSTN3D
        else:
            stn_model = AffineSTN2D
    else:
        raise NotImplementedError('STN {} not supported'.format(args.stn))

    print('STN: ' + str(stn_model))

    resampler_img = Resampler(config['spacing'], config['size'])
    resampler_seg = Resampler(config['spacing'], config['size'], is_label=True)

    if config['normalizer'] == 'zero_mean_unit_var':
        normalizer = Normalizer(zero_mean_unit_var)
    elif config['normalizer'] == 'range_matching':
        normalizer = Normalizer(range_matching)
    elif config['normalizer'] == 'zero_one':
        normalizer = Normalizer(zero_one)
    elif config['normalizer'] == 'threshold_zero':
        normalizer = Normalizer(threshold_zero)
    elif config['normalizer'] == 'none':
        normalizer = None
    else:
        raise NotImplementedError('Normalizer {} not supported'.format(config['normalizer']))

    stn_input_channels = 2 * (config['num_classes'] - 1)

    if args.mode3d:
        itn = UNet3D(num_classes=config['num_classes']).to(device)
    else:
        itn = UNet2D(num_classes=config['num_classes']).to(device)
    stn = stn_model(input_size=config['size'], input_channels=stn_input_channels, device=device).to(device)
    parameters = list(itn.parameters()) + list(stn.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])
    # set learning rate decay scheduler - decay to 50% every 'epoch_decay_steps' epochs
    gamma = 0.5 ** (1 / config['epoch_decay_steps'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    config_dict = {'config': config,
                   'device': device,
                   'normalizer': normalizer,
                   'resampler_img': resampler_img,
                   'resampler_seg': resampler_seg,
                   'stn': stn,
                   'itn': itn,
                   'optimizer': optimizer,
                   'scheduler': scheduler
                   }
    print('File config: {}'.format(config_dict))

    return AttrDict(config_dict)


def process_batch(config, batch_samples, atlas_img, atlas_lab, omega):
    image = batch_samples['image'].to(config.device)
    labelmap = batch_samples['labelmap'].to(config.device)

    atlas_image = torch.from_numpy(sitk.GetArrayFromImage(atlas_img))[None, None, ...].to(config.device)

    if len(atlas_image.size()) == 5:
        atlas_labelmap = torch.from_numpy(sitk.GetArrayFromImage(atlas_lab)).permute(3, 0, 1, 2).unsqueeze(0).to(config.device)
    else:
        atlas_labelmap = torch.from_numpy(sitk.GetArrayFromImage(atlas_lab)).permute(2, 0, 1).unsqueeze(0).to(config.device)

    repeats = np.ones(len(image.size()))
    repeats[0] = image.size(0)

    atlas_image = atlas_image.repeat(tuple(repeats.astype(int)))
    atlas_labelmap = atlas_labelmap.repeat(tuple(repeats.astype(int)))

    image_prime = config.itn(image)

    source = image_prime[:, 1::, ...]
    target = atlas_labelmap[:, 1::, ...]

    config.stn(torch.cat((source, target), dim=1))
    warped_image = config.stn.warp_inv_image(image)
    warped_image_prime = config.stn.warp_inv_image(image_prime)
    warped_labelmap = config.stn.warp_inv_image(labelmap)
    warped_atlas_image = config.stn.warp_image(atlas_image)
    warped_atlas_labelmap = config.stn.warp_image(atlas_labelmap)

    grid = mira_vis.make_grid_image(config.config['size'], 4, device=config.device)
    grid = grid.repeat(tuple(repeats.astype(int)))
    warp_img2atl = config.stn.warp_inv_image(grid, padding='zeros')
    warp_atl2img = config.stn.warp_image(grid, padding='zeros')

    labelmap_argmax = torch.argmax(labelmap, dim=1)
    image_prime_argmax = torch.argmax(image_prime, dim=1)
    warped_atlas_labelmap_argmax = torch.argmax(warped_atlas_labelmap, dim=1)

    #ISTN segmentation loss
    loss_itn2seg = F.mse_loss(image_prime, labelmap)

    #STN image losses
    loss_img2atl = F.mse_loss(warped_image, atlas_image)
    loss_atl2img = F.mse_loss(image, warped_atlas_image)

    #STN atlas losses
    loss_seg2atl = F.mse_loss(warped_labelmap[:, 1::, ...], atlas_labelmap[:, 1::, ...])
    loss_atl2seg = F.mse_loss(labelmap[:, 1::, ...], warped_atlas_labelmap[:, 1::, ...])

    #ITN atlas loss
    loss_itn2atl = F.mse_loss(warped_image_prime[:, 1::, ...], atlas_labelmap[:, 1::, ...])
    loss_atl2itn = F.mse_loss(image_prime[:, 1::, ...], warped_atlas_labelmap[:, 1::, ...])

    # Regularization term
    reg_weight = config.config['lambda']
    reg_term = config.stn.regularizer()

    loss_train = loss_itn2seg + omega * (loss_seg2atl + loss_atl2seg + reg_weight * reg_term)

    # Custom Metrics
    dice_itn = mira_metrics.dice_score(labelmap_argmax, image_prime_argmax, num_classes=config.config['num_classes'])
    dice_atl = mira_metrics.dice_score(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'])
    asd_atl = mira_metrics.average_surface_distance(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'], spacing=config.config['spacing'])
    hd_atl = mira_metrics.hausdorff_distance(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'], spacing=config.config['spacing'])
    prec_atl = mira_metrics.precision(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'])
    reca_atl = mira_metrics.recall(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'])

    values_dict = {'01_loss': loss_train.item(),
                   '02_loss_itn2seg': loss_itn2seg.item(),
                   '03_loss_img2atl': loss_img2atl.item(),
                   '04_loss_atl2img': loss_atl2img.item(),
                   '05_loss_seg2atl': loss_seg2atl.item(),
                   '06_loss_atl2seg': loss_atl2seg.item(),
                   '07_loss_itn2atl': loss_itn2atl.item(),
                   '08_loss_atl2itn': loss_atl2itn.item(),
                   '09_reg_term': reg_term.item(),
                   '10_metric_dice_itn': dice_itn[1::].tolist(),
                   '11_metric_dice_atl': dice_atl[1::].tolist(),
                   '12_metric_asd_atl': asd_atl[1::].tolist(),
                   '13_metric_hd_atl': hd_atl[1::].tolist(),
                   '14_metric_prec_atl': prec_atl[1::].tolist(),
                   '15_metric_reca_atl': reca_atl[1::].tolist()}

    images_dict = {'01_image': image,
                   '02_labelmap': labelmap,
                   '03_image_prime': image_prime,
                   '04_warped_atlas_image': warped_atlas_image,
                   '05_warped_atlas_labelmap': warped_atlas_labelmap,
                   '06_warp_atl2img': warp_atl2img,
                   '07_warped_image': warped_image,
                   '08_warped_labelmap': warped_labelmap,
                   '09_warped_image_prime': warped_image_prime,
                   '10_atlas_image': atlas_image,
                   '11_atlas_labelmap': atlas_labelmap,
                   '12_warp_img2atl': warp_img2atl}

    return loss_train, images_dict, values_dict


def process_batch_test(config, config_stn, batch_samples, atlas_img, atlas_lab):
    image = batch_samples['image'].to(config.device)
    labelmap = batch_samples['labelmap'].to(config.device)

    atlas_image = torch.from_numpy(sitk.GetArrayFromImage(atlas_img))[None, None, ...].to(config.device)

    if len(atlas_image.size()) == 5:
        atlas_labelmap = torch.from_numpy(sitk.GetArrayFromImage(atlas_lab)).permute(3, 0, 1, 2).unsqueeze(0).to(config.device)
    else:
        atlas_labelmap = torch.from_numpy(sitk.GetArrayFromImage(atlas_lab)).permute(2, 0, 1).unsqueeze(0).to(config.device)

    repeats = np.ones(len(image.size()))
    repeats[0] = image.size(0)

    atlas_image = atlas_image.repeat(tuple(repeats.astype(int)))
    atlas_labelmap = atlas_labelmap.repeat(tuple(repeats.astype(int)))

    image_prime = config.itn(image)

    source = image_prime[:, 1::, ...]
    target = atlas_labelmap[:, 1::, ...]

    config_stn.stn(torch.cat((source, target), dim=1))
    warped_image_prime = config_stn.stn.warp_inv_image(image_prime)
    warped_atlas_image = config_stn.stn.warp_image(atlas_image)
    warped_atlas_labelmap = config_stn.stn.warp_image(atlas_labelmap)

    transform = config_stn.stn.get_T()
    transform_inv = config_stn.stn.get_T_inv()

    labelmap_argmax = torch.argmax(labelmap, dim=1)
    atlas_labelmap_argmax = torch.argmax(atlas_labelmap, dim=1)
    image_prime_argmax = torch.argmax(image_prime, dim=1)
    warped_atlas_labelmap_argmax = torch.argmax(warped_atlas_labelmap, dim=1)

    #STN image losses
    loss_atl2img = F.mse_loss(image, warped_atlas_image)

    #ITN atlas loss
    loss_itn2atl = F.mse_loss(warped_image_prime[:, 1::, ...], atlas_labelmap[:, 1::, ...])
    loss_atl2itn = F.mse_loss(image_prime[:, 1::, ...], warped_atlas_labelmap[:, 1::, ...])

    # Regularization term
    reg_weight = config.config['lambda']
    reg_term = config_stn.stn.regularizer()

    loss_refine = loss_atl2itn + reg_weight * reg_term

    # Custom Metrics
    dice_id = mira_metrics.dice_score(labelmap_argmax, atlas_labelmap_argmax, num_classes=config.config['num_classes'])
    asd_id = mira_metrics.average_surface_distance(labelmap_argmax, atlas_labelmap_argmax, num_classes=config.config['num_classes'], spacing=config.config['spacing'])
    hd_id = mira_metrics.hausdorff_distance(labelmap_argmax, atlas_labelmap_argmax, num_classes=config.config['num_classes'], spacing=config.config['spacing'])
    prec_id = mira_metrics.precision(labelmap_argmax, atlas_labelmap_argmax, num_classes=config.config['num_classes'])
    reca_id = mira_metrics.recall(labelmap_argmax, atlas_labelmap_argmax, num_classes=config.config['num_classes'])

    dice_itn = mira_metrics.dice_score(labelmap_argmax, image_prime_argmax, num_classes=config.config['num_classes'])
    asd_itn = mira_metrics.average_surface_distance(labelmap_argmax, image_prime_argmax, num_classes=config.config['num_classes'], spacing=config.config['spacing'])
    hd_itn = mira_metrics.hausdorff_distance(labelmap_argmax, image_prime_argmax, num_classes=config.config['num_classes'], spacing=config.config['spacing'])
    prec_itn = mira_metrics.precision(labelmap_argmax, image_prime_argmax, num_classes=config.config['num_classes'])
    reca_itn = mira_metrics.recall(labelmap_argmax, image_prime_argmax, num_classes=config.config['num_classes'])

    dice_atl = mira_metrics.dice_score(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'])
    asd_atl = mira_metrics.average_surface_distance(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'], spacing=config.config['spacing'])
    hd_atl = mira_metrics.hausdorff_distance(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'], spacing=config.config['spacing'])
    prec_atl = mira_metrics.precision(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'])
    reca_atl = mira_metrics.recall(labelmap_argmax, warped_atlas_labelmap_argmax, num_classes=config.config['num_classes'])

    values_dict = {'01_loss': loss_refine.item(),
                   '02_reg_term': reg_term.item(),
                   '03_metric_dice_id': dice_id[1::].tolist(),
                   '04_metric_dice_itn': dice_itn[1::].tolist(),
                   '05_metric_dice_atl': dice_atl[1::].tolist(),
                   '06_metric_asd_id': asd_id[1::].tolist(),
                   '07_metric_asd_itn': asd_itn[1::].tolist(),
                   '08_metric_asd_atl': asd_atl[1::].tolist(),
                   '09_metric_hd_id': hd_id[1::].tolist(),
                   '10_metric_hd_itn': hd_itn[1::].tolist(),
                   '11_metric_hd_atl': hd_atl[1::].tolist(),
                   '12_metric_prec_id': prec_id[1::].tolist(),
                   '13_metric_prec_itn': prec_itn[1::].tolist(),
                   '14_metric_prec_atl': prec_atl[1::].tolist(),
                   '15_metric_reca_id': reca_id[1::].tolist(),
                   '16_metric_reca_itn': reca_itn[1::].tolist(),
                   '17_metric_reca_atl': reca_atl[1::].tolist()}

    images_dict = {'01_image': image,
                   '02_labelmap': labelmap,
                   '03_image_prime': image_prime,
                   '04_warped_atlas_image': warped_atlas_image,
                   '05_warped_atlas_labelmap': warped_atlas_labelmap,
                   '06_warped_image_prime': warped_image_prime,
                   '07_atlas_image': atlas_image,
                   '08_atlas_labelmap': atlas_labelmap,
                   '09_transform': transform,
                   '10_transform_inv': transform_inv}

    return loss_refine, images_dict, values_dict


def update_atlas(config, dataset, atlas_img, atlas_lab, alpha, init=False):
    config.itn.eval()
    config.stn.eval()

    atlas_image = torch.from_numpy(sitk.GetArrayFromImage(atlas_img))[None, None, ...].to(config.device)

    if len(atlas_image.size()) == 5:
        atlas_labelmap = torch.from_numpy(sitk.GetArrayFromImage(atlas_lab)).permute(3, 0, 1, 2).unsqueeze(0).to(config.device)
    else:
        atlas_labelmap = torch.from_numpy(sitk.GetArrayFromImage(atlas_lab)).permute(2, 0, 1).unsqueeze(0).to(config.device)

    atlas_image_update = torch.zeros(atlas_image.size()).to(config.device)
    atlas_labelmap_update = torch.zeros(atlas_labelmap.size()).to(config.device)

    with torch.no_grad():
        for idx, _ in enumerate(tqdm(range(len(dataset)), desc='Updating Atlas')):
            sample = dataset.get_sample(idx)
            image = torch.from_numpy(sitk.GetArrayFromImage(sample['image']))[None, None, ...].to(config.device)

            if len(image.size()) == 5:
                labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(3, 0, 1, 2).unsqueeze(0).to(config.device)
            else:
                labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(2, 0, 1).unsqueeze(0).to(config.device)

            if init:
                atlas_image_update += image
                atlas_labelmap_update += labelmap
            else:
                image_prime = config.itn(image)

                source = image_prime[:, 1::, ...]
                target = atlas_labelmap[:, 1::, ...]

                config.stn(torch.cat((source, target), dim=1))
                warped_image = config.stn.warp_inv_image(image)
                warped_labelmap = config.stn.warp_inv_image(labelmap)

                atlas_image_update += warped_image
                atlas_labelmap_update += warped_labelmap

    atlas_image_update /= len(dataset)
    atlas_labelmap_update /= len(dataset)

    atlas_image_update = (atlas_image * (1.0 - alpha) + atlas_image_update * alpha)
    atlas_labelmap_update = (atlas_labelmap * (1.0 - alpha) + atlas_labelmap_update * alpha)

    atlas_image_updated = sitk.GetImageFromArray(atlas_image_update.cpu().squeeze().detach().numpy())

    if len(atlas_image.size()) == 5:
        atlas_labelmap_updated = sitk.GetImageFromArray(atlas_labelmap_update.cpu().squeeze().detach().permute(1, 2, 3, 0).numpy(), isVector=True)
    else:
        atlas_labelmap_updated = sitk.GetImageFromArray(atlas_labelmap_update.cpu().squeeze().detach().permute(1, 2, 0).numpy(), isVector=True)


    atlas_image_updated.CopyInformation(atlas_img)
    atlas_labelmap_updated.CopyInformation(atlas_lab)

    return atlas_image_updated, atlas_labelmap_updated


def train(args):
    config = set_up_model_and_preprocessing('TRAINING', args)

    writer = SummaryWriter('{}/tensorboard'.format(args.out))
    global_step = 0

    print(separator)
    print('TRAINING data...')
    print(separator)

    dataset_train = ImageSegmentationOneHotDataset(args.train, args.train_seg, args.train_msk, normalizer=config.normalizer,
                                       resampler_img=config.resampler_img, resampler_seg=config.resampler_seg, binarize=config.config['binarize'], augmentation=False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.config['batch_size'], shuffle=True)

    if args.val is not None:
        print(separator)
        print('VALIDATION data...')
        print(separator)
        dataset_val = ImageSegmentationOneHotDataset(args.val, args.val_seg, args.val_msk, normalizer=config.normalizer,
                                         resampler_img=config.resampler_img, resampler_seg=config.resampler_seg, binarize=config.config['binarize'], augmentation=False)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    # Create output directory
    out_dir = os.path.join(args.out, 'train')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.save_temp:
        temp_dir = os.path.join(out_dir, 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        for idx in range(0, 5):
            sample = dataset_train.get_sample(idx)
            sitk.WriteImage(sample['image'], os.path.join(temp_dir, 'sample_' + str(idx) + '_image.nii.gz'))
            sitk.WriteImage(sample['labelmap'], os.path.join(temp_dir, 'sample_' + str(idx) + '_labelmap.nii.gz'))

    print(separator)

    # Note: Must match those used in process_batch()
    loss_names = ['01_loss', '02_loss_itn2seg', '03_loss_img2atl', '04_loss_atl2img', '05_loss_seg2atl', '06_loss_atl2seg', '07_loss_itn2atl', '08_loss_atl2itn', '09_reg_term', '10_metric_dice_itn', '11_metric_dice_atl', '12_metric_asd_atl', '13_metric_hd_atl', '14_metric_prec_atl', '15_metric_reca_atl']
    train_logger = mira_metrics.Logger('TRAIN', loss_names)
    validation_logger = mira_metrics.Logger('VALID', loss_names)

    model_dir = args.model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create initial atlas
    sample = dataset_train.get_sample(0)
    atlas_image = sample['image']
    atlas_labelmap = sample['labelmap']

    atlas_image, atlas_labelmap = update_atlas(config, dataset_train, atlas_image, atlas_labelmap, alpha=1.0, init=True)

    sitk.WriteImage(atlas_image, model_dir + '/atlas_image_initial.nii.gz')
    sitk.WriteImage(atlas_labelmap, model_dir + '/atlas_labelmap_initial.nii.gz')

    for epoch in range(1, config.config['epochs'] + 1):
        config.itn.train()
        config.stn.train()

        if config.config['epoch_loss_fading'] != -1:
            omega = 1 / (1 + np.exp(-(epoch - config.config['epoch_loss_fading']) / 25))
        else:
            omega = 1

        # Training
        for batch_idx, batch_samples in enumerate(tqdm(dataloader_train, desc='Epoch {}'.format(epoch))):
            global_step += 1
            config.optimizer.zero_grad()
            loss, images_dict, values_dict = process_batch(config, batch_samples, atlas_image, atlas_labelmap, omega)
            loss.backward()
            config.optimizer.step()
            train_logger.update_epoch_logger(values_dict)

        # iterate learning rate decay
        if config.config['epoch_decay_steps']:
            config.scheduler.step()

        train_logger.update_epoch_summary(epoch)
        write_values(writer, 'train', value_dict=train_logger.get_latest_dict(), n_iter=global_step)
        write_images(writer, 'train', image_dict=images_dict, n_iter=global_step, mode3d=args.mode3d)

        # Validation
        if args.val is not None and (epoch == 1 or epoch % config.config['val_interval'] == 0):
            config.itn.eval()
            config.stn.eval()

            with torch.no_grad():
                for batch_idx, batch_samples in enumerate(dataloader_val):
                    loss, images_dict, values_dict = process_batch(config, batch_samples, atlas_image, atlas_labelmap, omega)
                    validation_logger.update_epoch_logger(values_dict)

            validation_logger.update_epoch_summary(epoch)
            write_values(writer, phase='val', value_dict=validation_logger.get_latest_dict(), n_iter=global_step)
            write_images(writer, phase='val', image_dict=images_dict, n_iter=global_step, mode3d=args.mode3d)

            print(separator)
            train_logger.print_latest()
            validation_logger.print_latest()
            print(separator)

            torch.save(config.itn.state_dict(), model_dir + '/itn_' + str(epoch) + '.pt')
            torch.save(config.stn.state_dict(), model_dir + '/stn_' + str(epoch) + '.pt')

        # Update atlas
        atlas_image, atlas_labelmap = update_atlas(config, dataset_train, atlas_image, atlas_labelmap, alpha=config.config['alpha'])

    torch.save(config.itn.state_dict(), model_dir + '/itn.pt')
    torch.save(config.stn.state_dict(), model_dir + '/stn.pt')

    sitk.WriteImage(atlas_image, model_dir + '/atlas_image_final.nii.gz')
    sitk.WriteImage(atlas_labelmap, model_dir + '/atlas_labelmap_final.nii.gz')

    print(separator)
    print('Finished TRAINING... Plotting Graphs\n\n')
    for loss_name, colour in zip(['01_loss'], ['b']):
        plt.plot(train_logger.epoch_number_logger, train_logger.epoch_summary[loss_name], c=colour,
                 label='train {}'.format(loss_name))
        plt.plot(validation_logger.epoch_number_logger, validation_logger.epoch_summary[loss_name], c=colour,
                 linestyle=':',
                 label='val {}'.format(loss_name))

    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def test(args):
    config = set_up_model_and_preprocessing('TESTING', args)

    dataset_test = ImageSegmentationOneHotDataset(args.test, args.test_seg, args.test_msk, normalizer=config.normalizer,
                                      resampler_img=config.resampler_img, resampler_seg=config.resampler_seg, binarize=config.config['binarize'], augmentation=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)
    loss_names = ['01_loss', '02_reg_term', '03_metric_dice_id', '04_metric_dice_itn', '05_metric_dice_atl', '06_metric_asd_id', '07_metric_asd_itn', '08_metric_asd_atl', '09_metric_hd_id', '10_metric_hd_itn', '11_metric_hd_atl', '12_metric_prec_id', '13_metric_prec_itn', '14_metric_prec_atl', '15_metric_reca_id', '16_metric_reca_itn', '17_metric_reca_atl']
    test_logger = mira_metrics.Logger('TEST', loss_names)

    # Create output directory
    out_dir = os.path.join(args.out, 'test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load atlas
    atlas_image = sitk.ReadImage(args.model + '/atlas_image_final.nii.gz')
    atlas_labelmap = sitk.ReadImage(args.model + '/atlas_labelmap_final.nii.gz')

    config.itn.load_state_dict(torch.load(args.model + '/itn.pt'))
    config.itn.eval()

    config.stn.load_state_dict(torch.load(args.model + '/stn.pt'))
    config.stn.eval()

    with torch.no_grad():
        for index, batch_samples in enumerate(dataloader_test):
            loss, images_dict, values_dict = process_batch_test(config, config, batch_samples, atlas_image, atlas_labelmap)
            test_logger.update_epoch_logger(values_dict)

            image = sitk.GetImageFromArray(images_dict['01_image'].cpu().squeeze().numpy())
            image.CopyInformation(dataset_test.get_sample(index)['image'])
            sitk.WriteImage(image,
                            os.path.join(out_dir, 'sample_' + str(index) + '_image.nii.gz'))

            if args.mode3d:
                warped_atlas_labelmap = sitk.GetImageFromArray(images_dict['05_warped_atlas_labelmap'].cpu().squeeze().detach().permute(1, 2, 3, 0).numpy(), isVector=True)
                image_prime = sitk.GetImageFromArray(images_dict['03_image_prime'].cpu().squeeze().detach().permute(1, 2, 3, 0).numpy(), isVector=True)
                labelmap = sitk.GetImageFromArray(images_dict['02_labelmap'].cpu().squeeze().detach().permute(1, 2, 3, 0).numpy(), isVector=True)
            else:
                warped_atlas_labelmap = sitk.GetImageFromArray(images_dict['05_warped_atlas_labelmap'].cpu().squeeze().detach().permute(1, 2, 0).numpy(), isVector=True)
                image_prime = sitk.GetImageFromArray(images_dict['03_image_prime'].cpu().squeeze().detach().permute(1, 2, 0).numpy(), isVector=True)
                labelmap = sitk.GetImageFromArray(images_dict['02_labelmap'].cpu().squeeze().detach().permute(1, 2, 0).numpy(), isVector=True)

            warped_atlas_labelmap_argmax = sitk.GetImageFromArray(torch.argmax(images_dict['05_warped_atlas_labelmap'], dim=1).cpu().squeeze().detach().numpy().astype(np.float32))
            image_prime_argmax = sitk.GetImageFromArray(torch.argmax(images_dict['03_image_prime'], dim=1).cpu().squeeze().detach().numpy().astype(np.float32))
            labelmap_argmax = sitk.GetImageFromArray(torch.argmax(images_dict['02_labelmap'], dim=1).cpu().squeeze().detach().numpy().astype(np.float32))

            transform = sitk.GetImageFromArray(images_dict['09_transform'].cpu().squeeze().detach().numpy(),
                                               isVector=True)
            transform_inv = sitk.GetImageFromArray(images_dict['10_transform_inv'].cpu().squeeze().detach().numpy(),
                                                   isVector=True)

            warped_atlas_labelmap.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(warped_atlas_labelmap,
                            os.path.join(out_dir, 'sample_' + str(index) + '_warped_atlas_labelmap.nii.gz'))

            warped_atlas_labelmap_argmax.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(warped_atlas_labelmap_argmax,
                            os.path.join(out_dir, 'sample_' + str(index) + '_warped_atlas_labelmap_argmax.nii.gz'))

            image_prime.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(image_prime,
                            os.path.join(out_dir, 'sample_' + str(index) + '_image_prime.nii.gz'))

            image_prime_argmax.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(image_prime_argmax,
                            os.path.join(out_dir, 'sample_' + str(index) + '_image_prime_argmax.nii.gz'))

            labelmap.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(labelmap,
                            os.path.join(out_dir, 'sample_' + str(index) + '_labelmap.nii.gz'))

            labelmap_argmax.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(labelmap_argmax,
                            os.path.join(out_dir, 'sample_' + str(index) + '_labelmap_argmax.nii.gz'))

            transform.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(transform,
                            os.path.join(out_dir, 'sample_' + str(index) + '_transform.nii.gz'))

            transform_inv.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(transform_inv,
                            os.path.join(out_dir, 'sample_' + str(index) + '_transform_inv.nii.gz'))

        with open(os.path.join(out_dir,'test_results.yml'), 'w') as outfile:
            yaml.dump(test_logger.get_epoch_logger(), outfile)
    test_logger.update_epoch_summary(0)

    if args.refine == True:
        refine_config = set_up_model_and_preprocessing('REFINEMENT', args)
        config.itn.eval()

        for index, batch_samples in enumerate(dataloader_test):

            print('Processing image ' + str(index+1) + ' of ' + str(len(dataset_test)))

            refine_config.stn.load_state_dict(torch.load(args.model + '/stn.pt'))
            refine_config.stn.train()

            parameters = list(refine_config.stn.parameters())
            optimizer = torch.optim.Adam(parameters, lr=config.config['learning_rate'])

            # Fine tune STN
            for epoch in range(1, config.config['refine'] + 1):
                optimizer.zero_grad()
                loss, images_dict, values_dict = process_batch_test(config, refine_config, batch_samples, atlas_image, atlas_labelmap)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                loss, images_dict, values_dict = process_batch_test(config, refine_config, batch_samples, atlas_image, atlas_labelmap)
                test_logger.update_epoch_logger(values_dict)

                if args.mode3d:
                    warped_atlas_labelmap = sitk.GetImageFromArray(images_dict['05_warped_atlas_labelmap'].cpu().squeeze().detach().permute(1, 2, 3, 0).numpy(), isVector=True)
                else:
                    warped_atlas_labelmap = sitk.GetImageFromArray(images_dict['05_warped_atlas_labelmap'].cpu().squeeze().detach().permute(1, 2, 0).numpy(), isVector=True)

                transform = sitk.GetImageFromArray(images_dict['09_transform'].cpu().squeeze().detach().numpy(),
                                                   isVector=True)
                transform_inv = sitk.GetImageFromArray(images_dict['10_transform_inv'].cpu().squeeze().detach().numpy(),
                                                       isVector=True)

                warped_atlas_labelmap_argmax = sitk.GetImageFromArray(
                    torch.argmax(images_dict['05_warped_atlas_labelmap'], dim=1).cpu().squeeze().detach().numpy().astype(np.float32))

                warped_atlas_labelmap.CopyInformation(dataset_test.get_sample(index)['labelmap'])
                sitk.WriteImage(warped_atlas_labelmap,
                                os.path.join(out_dir, 'sample_' + str(index) + '_warped_atlas_labelmap_refined.nii.gz'))

                warped_atlas_labelmap_argmax.CopyInformation(dataset_test.get_sample(index)['labelmap'])
                sitk.WriteImage(warped_atlas_labelmap_argmax,
                                os.path.join(out_dir, 'sample_' + str(index) + '_warped_atlas_labelmap_argmax_refined.nii.gz'))

                transform.CopyInformation(dataset_test.get_sample(index)['labelmap'])
                sitk.WriteImage(transform,
                                os.path.join(out_dir, 'sample_' + str(index) + '_transform_refined.nii.gz'))

                transform_inv.CopyInformation(dataset_test.get_sample(index)['labelmap'])
                sitk.WriteImage(transform_inv,
                                os.path.join(out_dir, 'sample_' + str(index) + '_transform_inv_refined.nii.gz'))

            with open(os.path.join(out_dir, 'test_results_refined.yml'), 'w') as outfile:
                yaml.dump(test_logger.get_epoch_logger(), outfile)


if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='atlas segmentation')
    parser.add_argument('--save_temp', default=True, action='store_true', help='save temporary files (default: True)')
    parser.add_argument('--dev', default='0', help='cuda device (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')


    # SYNTH 2D
    #
    # Data args
    parser.add_argument('--train', default='data/synth2d/train.csv', help='training data csv file')
    parser.add_argument('--train_seg', default='data/synth2d/train.seg.csv', help='training data csv file')
    parser.add_argument('--train_msk', default=None, help='training data csv file')
    parser.add_argument('--val', default='data/synth2d/val.csv', help='validation data csv file')
    parser.add_argument('--val_seg', default='data/synth2d/val.seg.csv', help='validation data csv file')
    parser.add_argument('--val_msk', default=None, help='validation data csv file')
    parser.add_argument('--test', default='data/synth2d/test.corrupt.csv', help='testing data csv file')
    parser.add_argument('--test_seg', default='data/synth2d/test.corrupt.seg.csv', help='testing data csv file')
    # parser.add_argument('--test', default='data/synth2d/test.example.img.csv', help='testing data csv file')
    # parser.add_argument('--test_seg', default='data/synth2d/test.example.seg.csv', help='testing data csv file')
    parser.add_argument('--test_msk', default=None, help='testing data csv file')

    # Network args
    parser.add_argument('--mode3d', default=False, action='store_true', help='enable 3D mode', )
    parser.add_argument('--config', default="data/synth2d/config.json", help='config file')

    # Logging args
    parser.add_argument('--out', default='output/synth2d/full-stn', help='output root directory')
    parser.add_argument('--model', default='output/synth2d/full-stn/train/model', help='model directory')


    parser.add_argument('--stn', default="f",
                        help='stn type, f=full, s=svf, a=affine',
                        choices=['f', 's', 'a'])
    parser.add_argument('--refine', default=True, action='store_true', help='enable iterative refinement', )

    args = parser.parse_args()

    # Run training
    if args.train is not None:
        train(args)

    # Run testing
    if args.test is not None:
        test(args)
