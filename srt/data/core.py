import numpy as np
import os
from srt import data


def get_dataset(mode, cfg, phase1=True):
    ''' Returns a dataset.
    Args:
        mode: Dataset split, 'train' or 'val'
        cfg (dict): data config dictionary
    '''
    dataset_type = cfg['dataset']

    # Create dataset
    if dataset_type == 'vox':
        dataset = data.VoxDataset(path=cfg['data_path'],
                                  kp_path=cfg['kp_path'],
                                  mode=mode,
                                  phase1=phase1,
                                  num_src=cfg['num_src'],
                                  num_pixels=cfg['num_pixels'],
                                  num_pixels_phase1=cfg['num_pixels_phase1'],
                                  augmentation_params=cfg['augmentation_params'],
                                  image_subsampling=cfg['image_subsampling'],
                                  simulate_out_of_frame_motion=cfg['simulate_out_of_frame_motion'])
    else:
        raise ValueError('Invalid dataset "{}"'.format(cfg['dataset']))

    return dataset


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

