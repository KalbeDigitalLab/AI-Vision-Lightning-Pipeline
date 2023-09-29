import os
import random

import deeplake
import numpy as np
import pytest


@pytest.fixture(scope='session')
def dummy_train_dataset(tmp_path_factory):
    label_img = ['abdominal', 'adult', 'pediatric', 'spine', 'others']

    root_dir = tmp_path_factory.mktemp('dummy_bodypartxr_dataset_train')

    num_samples = 50
    ds = deeplake.dataset(root_dir)
    with ds:
        ds.create_tensor('images', htype='image',
                         sample_compression='jpeg')
        ds.create_tensor('labels', htype='class_label',
                         class_names=label_img)

    with ds:
        for i in range(num_samples):
            dummy_label = random.choice(label_img)  # nosec
            label_num = label_img.index(dummy_label)

            ds.append({
                'images': np.random.randint(0, 256, size=(512, 512), dtype='uint8'),
                'labels': np.uint32(label_num)
            })

    return str(root_dir)


@pytest.fixture(scope='session')
def dummy_val_dataset(tmp_path_factory):
    label_img = ['abdominal', 'adult', 'pediatric', 'spine', 'others']

    root_dir = tmp_path_factory.mktemp('dummy_bodypartxr_dataset_val')
# image, label

    num_samples = 30
    ds = deeplake.dataset(root_dir)
    with ds:
        ds.create_tensor('images', htype='image',
                         sample_compression='jpeg')
        ds.create_tensor('labels', htype='class_label',
                         class_names=label_img)

    with ds:
        for i in range(num_samples):
            dummy_label = random.choice(label_img)  # nosec
            label_num = label_img.index(dummy_label)

            ds.append({
                'images': np.random.randint(0, 256, size=(512, 512), dtype='uint8'),
                'labels': np.uint32(label_num)
            })

    return str(root_dir)


@pytest.fixture(scope='session')
def dummy_test_dataset(tmp_path_factory):
    label_img = ['abdominal', 'adult', 'pediatric', 'spine', 'others']

    root_dir = tmp_path_factory.mktemp('dummy_bodypartxr_dataset_test')

    num_samples = 25
    ds = deeplake.dataset(root_dir)
    with ds:
        ds.create_tensor('images', htype='image',
                         sample_compression='jpeg')
        # Classification
        ds.create_tensor('labels', htype='class_label',
                         class_names=label_img)

    with ds:
        for i in range(num_samples):
            dummy_label = random.choice(label_img)  # nosec
            label_num = label_img.index(dummy_label)

            ds.append({
                'images': np.random.randint(0, 256, size=(512, 512), dtype='uint8'),
                'labels': np.uint32(label_num)
            })

    return str(root_dir)
