import os
import random

import cv2
import fiftyone
import numpy as np
import pytest
import xxhash


@pytest.fixture(scope='session')
def vindr_mammography_dataset_dir(tmp_path_factory):
    birads_level = [f'BI-RADS {i}' for i in range(6)]
    density_level = [f'DENSITY {i}' for i in ['A', 'B', 'C', 'D']]

    root_dir = tmp_path_factory.mktemp('temp_mammography_vindr_dataset')

    dataset = fiftyone.Dataset()
    num_samples = 40
    for i in range(num_samples):
        group = fiftyone.Group()
        study_id = xxhash.xxh64().hexdigest()
        samples = []
        split = random.choice(['train', 'test'])  # nosec B311

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
            breast_density = ''
            breast_birads = ''
            laterality = ''
            view_position = ''

            laterality, view_position = side_view.split('_')
            breast_birads = random.choice(birads_level)  # nosec B311
            breast_density = random.choice(density_level)  # nosec B311
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
