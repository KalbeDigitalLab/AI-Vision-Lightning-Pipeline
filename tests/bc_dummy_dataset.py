import os
import random

import deeplake
import numpy as np
import pytest


@pytest.fixture(scope='session')
def deeplake_dummy_dataset_dir(tmp_path_factory):
    """Fixture to create a Deeplake dummy dataset directory for testing purposes.

    This fixture generates a temporary Deeplake dataset directory with dummy image and label data
    for use in testing. It creates 'train', 'val', and 'test' splits containing random image and label data.

    Parameters:
        tmp_path_factory (TempPathFactory): A temporary path factory provided by pytest.

    Returns:
        str: The path to the generated dummy dataset directory.
    """
    label_img = ['normal', 'benign', 'malignant']

    root_dir = tmp_path_factory.mktemp('temp_deeplake_dummy_dataset')

    num_samples = 30
    for split in ['train', 'val', 'test']:
        output_dir = os.path.join(str(root_dir), f'dummy-{split}')
        print(output_dir)
        ds = deeplake.dataset(output_dir)
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
                    'images': np.random.randint(0, 256, size=(256, 256), dtype='uint8'),
                    'labels': np.uint32(label_num)
                })

    return str(root_dir)


@pytest.fixture(scope='session')
def deeplake_dummy_train_dataset_dir(tmp_path_factory):
    """Fixture to create a Deeplake dummy training dataset directory for testing purposes.

    This fixture generates a temporary Deeplake dataset directory for training data with dummy image and label data.

    Parameters:
        tmp_path_factory (TempPathFactory): A temporary path factory provided by pytest.

    Returns:
        str: The path to the generated dummy training dataset directory.
    """
    label_img = ['normal', 'benign', 'malignant']

    root_dir = tmp_path_factory.mktemp('temp_bc_dummy_dataset_train')

    num_samples = 547
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
                'images': np.random.randint(0, 256, size=(256, 256), dtype='uint8'),
                'labels': np.uint32(label_num)
            })

    return str(root_dir)


@pytest.fixture(scope='session')
def deeplake_dummy_val_dataset_dir(tmp_path_factory):
    """Fixture to create a Deeplake dummy validation dataset directory for testing purposes.

    This fixture generates a temporary Deeplake dataset directory for validation data with dummy image and label data.

    Parameters:
        tmp_path_factory (TempPathFactory): A temporary path factory provided by pytest.

    Returns:
        str: The path to the generated dummy validation dataset directory.
    """
    label_img = ['normal', 'benign', 'malignant']

    root_dir = tmp_path_factory.mktemp('temp_bc_dummy_dataset_val')

    num_samples = 233
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
                'images': np.random.randint(0, 256, size=(256, 256), dtype='uint8'),
                'labels': np.uint32(label_num)
            })

    return str(root_dir)


@pytest.fixture(scope='session')
def deeplake_dummy_test_dataset_dir(tmp_path_factory):
    """Fixture to create a Deeplake dummy test dataset directory for testing purposes.

    This fixture generates a temporary Deeplake dataset directory for test data with dummy image and label data.

    Parameters:
        tmp_path_factory (TempPathFactory): A temporary path factory provided by pytest.

    Returns:
        str: The path to the generated dummy test dataset directory.
    """
    label_img = ['normal', 'benign', 'malignant']

    root_dir = tmp_path_factory.mktemp('temp_bc_dummy_dataset_test')

    num_samples = 155
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
                'images': np.random.randint(0, 256, size=(256, 256), dtype='uint8'),
                'labels': np.uint32(label_num)
            })

    return str(root_dir)
