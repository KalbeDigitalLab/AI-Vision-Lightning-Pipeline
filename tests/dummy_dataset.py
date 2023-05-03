import os
import random

import cv2
import fiftyone
import numpy as np
import pytest
import torch
import xxhash


@pytest.fixture(scope='session')
def vindr_4views_mammography_single_dataset_dir(tmp_path_factory):
    birads_level = [f'BI-RADS {i}' for i in range(6)]
    density_level = [f'DENSITY {i}' for i in ['A', 'B', 'C', 'D']]

    root_dir = tmp_path_factory.mktemp('temp_mammography_4views_vindr_dataset')

    dataset = fiftyone.Dataset()
    num_samples = 40
    for split in ['train', 'test']:
        for i in range(num_samples):
            group = fiftyone.Group()
            study_id = xxhash.xxh64().hexdigest()
            samples = []
            breast_birads = random.choice(birads_level)  # nosec B311
            breast_density = random.choice(density_level)  # nosec B311

            for g_id, side_view in enumerate(['left_cc', 'left_mlo', 'right_cc', 'right_mlo']):
                # Save to PNG format

                os.makedirs(os.path.join(str(root_dir), 'images', study_id), exist_ok=True)
                png_file_path = os.path.join(str(root_dir), 'images', study_id, str(g_id)+'.png')
                image = np.random.randint(0, 256, size=(384, 384))
                cv2.imwrite(png_file_path, image)  # lossless

                # Build metadata
                metadata = fiftyone.ImageMetadata.build_for(png_file_path)
                sample = fiftyone.Sample(
                    filepath=png_file_path,
                    metadata=metadata,
                )

                tags = []
                detections = []
                laterality = ''
                view_position = ''

                laterality, view_position = side_view.split('_')
                tags += [split, laterality, view_position]

                # Remove duplicates and add to sample tags
                tags = set(tags)
                for tag in tags:
                    sample.tags.append(tag)

                # Grouped sample
                sample['view_side'] = group.element(f'{laterality}_{view_position}')
                sample['laterality'] = laterality
                sample['view_position'] = view_position
                sample['breast_birads'] = fiftyone.Classification(label=breast_birads)
                sample['breast_density'] = fiftyone.Classification(label=breast_density)
                sample['findings_objects'] = fiftyone.Detections(detections=detections)
                sample['study_id'] = study_id

                samples.append(sample)

            dataset.add_samples(samples)

    output_dir = os.path.join(str(root_dir), 'temp_group_dataset')
    dataset.export(
        export_dir=output_dir,
        dataset_type=fiftyone.types.FiftyOneDataset,
        export_media=True
    )

    return output_dir


@pytest.fixture(scope='session')
def vindr_2views_mammography_single_dataset_dir(tmp_path_factory):
    birads_level = [f'BI-RADS {i}' for i in range(6)]
    density_level = [f'DENSITY {i}' for i in ['A', 'B', 'C', 'D']]

    root_dir = tmp_path_factory.mktemp('temp_mammography_2views_vindr_dataset')

    dataset = fiftyone.Dataset()
    num_samples = 40

    for split in ['train', 'test']:
        for i in range(num_samples):
            group = fiftyone.Group()
            study_id = xxhash.xxh64().hexdigest()
            samples = []
            laterality = random.choice(['left', 'right'])  # nosec B311
            breast_birads = random.choice(birads_level)  # nosec B311
            breast_density = random.choice(density_level)  # nosec B311

            for g_id, side_view in enumerate(['cc', 'mlo']):
                # Save to PNG format

                os.makedirs(os.path.join(str(root_dir), 'images', study_id), exist_ok=True)
                png_file_path = os.path.join(str(root_dir), 'images', study_id, str(g_id)+'.png')
                image = np.random.randint(0, 256, size=(384, 384))
                cv2.imwrite(png_file_path, image)  # lossless

                # Build metadata
                metadata = fiftyone.ImageMetadata.build_for(png_file_path)
                sample = fiftyone.Sample(
                    filepath=png_file_path,
                    metadata=metadata,
                )

                tags = []
                detections = []

                view_position = side_view
                tags += [split, laterality, side_view]

                # Remove duplicates and add to sample tags
                tags = set(tags)
                for tag in tags:
                    sample.tags.append(tag)

                # Grouped sample
                sample['view_side'] = group.element(f'{laterality}_{view_position}')
                sample['laterality'] = laterality
                sample['view_position'] = view_position
                sample['breast_birads'] = fiftyone.Classification(label=breast_birads)
                sample['breast_density'] = fiftyone.Classification(label=breast_density)
                sample['findings_objects'] = fiftyone.Detections(detections=detections)
                sample['study_id'] = study_id

                samples.append(sample)

            dataset.add_samples(samples)

    output_dir = os.path.join(str(root_dir), 'temp_group_dataset')
    dataset.export(
        export_dir=output_dir,
        dataset_type=fiftyone.types.FiftyOneDataset,
        export_media=True
    )

    return output_dir


@pytest.fixture(scope='session')
def empty_dataset_dir(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp('temp_empty_dataset')

    dataset = fiftyone.Dataset()
    output_dir = os.path.join(str(root_dir), 'temp_group_dataset')
    dataset.export(
        export_dir=output_dir,
        dataset_type=fiftyone.types.FiftyOneDataset,
        export_media=True
    )

    return output_dir


@pytest.fixture(scope='session')
def inbreast_2views_patches_dataset_path(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp('temp_inbreast_2views_vindr_dataset')

    num_samples = 40

    samples = []
    for _ in range(num_samples):
        image = torch.randint(0, 256, (2, 384, 384)).to(dtype=torch.float32)
        label = random.choice([0, 1, 2, 3, 4])  # nosec B311
        samples.append((image, label))

    output_dir = os.path.join(str(root_dir), 'temp_inbreast_2views_dataset_path.pt')
    torch.save(samples, output_dir)

    return output_dir


@pytest.fixture(scope='session')
def inbreast_2views_single_dataset_path(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp('temp_inbreast_2views_vindr_dataset')

    num_samples = 40

    samples = []
    for _ in range(num_samples):
        image = torch.randint(0, 256, (2, 384, 384)).to(dtype=torch.float32)
        label = random.choice([0, 1])  # nosec B311
        samples.append((image, label))

    output_dir = os.path.join(str(root_dir), 'temp_inbreast_2views_dataset_path.pt')
    torch.save(samples, output_dir)

    return output_dir


@pytest.fixture(scope='session')
def inbreast_2views_multi_dataset_path(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp('temp_inbreast_2views_vindr_dataset')

    num_samples = 40

    samples = []
    for _ in range(num_samples):
        image = torch.randint(0, 256, (2, 384, 384)).to(dtype=torch.float32)
        label = random.choice([[0, 1], [1, 0]])  # nosec B311
        samples.append((image, label))

    output_dir = os.path.join(str(root_dir), 'temp_inbreast_2views_dataset_path.pt')
    torch.save(samples, output_dir)

    return output_dir


@pytest.fixture(scope='session')
def inbreast_2views_multi_dataset_dir(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp('temp_inbreast_2views_vindr_dataset_dir')

    num_samples = 40

    for num_samples, name in [(40, 'train'), (96, 'validation')]:
        samples = []
        for _ in range(num_samples):
            image = torch.randint(0, 256, (2, 384, 384)).to(dtype=torch.float32)
            label = random.choice([[0, 1], [1, 0]])  # nosec B311
            samples.append((image, label))

        output_dir = os.path.join(str(root_dir), f'temp_{name}inbreast_2views_dataset.pt')
        torch.save(samples, output_dir)

    return str(root_dir)


@pytest.fixture(scope='session')
def inbreast_4views_multi_dataset_dir(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp('temp_inbreast_2views_vindr_dataset_dir')

    num_samples = 40

    for num_samples, name in [(40, 'train'), (96, 'validation')]:
        samples = []
        for _ in range(num_samples):
            image = torch.randint(0, 256, (4, 384, 384)).to(dtype=torch.float32)
            label = random.choice([[0, 1], [1, 0]])  # nosec B311
            samples.append((image, label))

        output_dir = os.path.join(str(root_dir), f'temp_{name}inbreast_2views_dataset.pt')
        torch.save(samples, output_dir)

    return str(root_dir)
