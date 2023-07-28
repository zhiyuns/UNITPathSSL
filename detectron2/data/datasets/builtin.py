# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_NUCLEUS = {}
_PREDEFINED_SPLITS_NUCLEUS["lizard"] = {
    "lizard_dpath_train_18": ("lizard_dataset/img2021_resized", "lizard_dataset/dpath.json"),
    # randomized, for GAN evaluation
    "lizard_dpath_train_8": ("lizard_dataset/img2021_resized", "lizard_dataset/dpath_train_40x_8.json"),  # randomized
    "lizard_dpath_train_4": ("lizard_dataset/img2021_resized", "lizard_dataset/dpath_train_40x_4.json"),  # randomized
    "lizard_dpath_train_2": ("lizard_dataset/img2021_resized", "lizard_dataset/dpath_train_40x_2.json"),  # randomized
    "lizard_dpath_train_extreme_small": ("lizard_dataset/img2021_resized", "lizard_dataset/dpath_train_40x_extremesmall.json"),  # 69-0-0~5
    "lizard_dpath_val_17": ("lizard_dataset/img2021_40x", "lizard_dataset/dpath_val_40x.json"),  # randomized
    "lizard_dpath_test_17": ("lizard_dataset/img2021_40x", "lizard_dataset/dpath_test_40x.json"),  # randomized
    "lizard_crag_train_32": ("lizard_dataset/img2021_40x", "lizard_dataset/crag_train_40x_32.json"),  # crag 33-64
    "lizard_crag_train_16": ("lizard_dataset/img2021_40x", "lizard_dataset/crag_train_40x_16.json"),  # crag 33-48
    "lizard_crag_train_8": ("lizard_dataset/img2021_40x", "lizard_dataset/crag_train_40x_8.json"),  # crag 33-40
    "lizard_crag_train_2": ("lizard_dataset/img2021_40x", "lizard_dataset/crag_train_40x_2.json"),  # crag 33-35
    "lizard_crag_train_1": ("lizard_dataset/img2021_40x", "lizard_dataset/crag_train_40x_1.json"),  # crag 33
    "lizard_crag_val_16": ("lizard_dataset/img2021_40x", "lizard_dataset/crag_val_40x.json"),  # crag 1-15
    "lizard_crag_test_16": ("lizard_dataset/img2021_40x", "lizard_dataset/crag_test_40x.json"),  # crag 16-31
    "lizard_crag_train_20x_8": ("lizard_dataset/img2021_20x", "lizard_dataset/crag_train_20x_8.json"),  # crag 33-40
    "lizard_crag_val_20x_16": ("lizard_dataset/img2021_20x", "lizard_dataset/crag_val_20x.json"),  # crag 1-16
}

_PREDEFINED_SPLITS_NUCLEUS["inhouse"] = {
    "inhouse_colon_train": ("inhouse_colon/img2021", "inhouse_colon/train.json"),
    # id: "7577750", "24393502"
    "inhouse_colon_val": ("inhouse_colon/img2021", "inhouse_colon/val.json"),
    # id: "7691016", "12278455"
}

_PREDEFINED_SPLITS_NUCLEUS["kumar"] = {
    "monuseg_colon_test": ("monuseg_colon/test/img2021", "monuseg_colon/test/instances_monuseg.json"),
    # only 3 images
    "monuseg_train": ("monuseg/img_40x", "monuseg/monuseg_train.json"),
    "monuseg_test": ("monuseg/img_40x", "monuseg/monuseg_test.json"),
    "monuseg_breast_train": ("monuseg_breast/img_cropped", "monuseg_breast/breast_train.json"),
    "monuseg_breast_test": ("monuseg_breast/img_cropped", "monuseg_breast/breast_test.json"),
    'kumar_train': ("kumar/img_40x", "kumar/train.json"),
    'kumar_train_cropped': ("kumar/img_40x_cropped", "kumar/train_cropped.json"),
    'kumar_test_diff': ("kumar/img_40x", "kumar/test_diff.json"),
    'kumar_test_same': ("kumar/img_40x", "kumar/test_same.json"),
    'kumar_test_diff_panoptic': ("kumar/img_40x", "kumar/test_diff_p.json"),
    'kumar_test_same_panoptic': ("kumar/img_40x", "kumar/test_same_p.json"),
    'kumar_test_panoptic': ("kumar/img_40x", "kumar/test_p.json"),
}

_PREDEFINED_SPLITS_NUCLEUS["TNBC"] = {
    "tnbc_all": ("TNBC/image", "TNBC/all.json"),
}

_PREDEFINED_SPLITS_NUCLEUS_MULTI = {}
_PREDEFINED_SPLITS_NUCLEUS_MULTI["lizard_multiclass"] = {
    "lizard_dpath_train_18_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/dpath.json"),
    # randomized, for GAN evaluation
    "lizard_dpath_train_8_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/dpath_train_40x_8.json"),  # randomized
    "lizard_dpath_train_4_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/dpath_train_40x_4.json"),  # randomized
    "lizard_dpath_train_2_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/dpath_train_40x_2.json"),  # randomized
    "lizard_dpath_train_1_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/dpath_train_40x_1.json"),
    "lizard_dpath_val_17_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/dpath_val_40x.json"),  # randomized
    "lizard_dpath_test_17_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/dpath_test_40x.json"),  # randomized
    "lizard_crag_train_32_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/crag_train_40x_32.json"),  # crag 33-64
    "lizard_crag_train_16_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/crag_train_40x_16.json"),  # crag 33-48
    "lizard_crag_train_8_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/crag_train_40x_8.json"),  # crag 33-40
    "lizard_crag_train_2_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/crag_train_40x_2.json"),  # crag 33-35
    "lizard_crag_train_1_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/crag_train_40x_1.json"),  # crag 33
    "lizard_crag_val_16_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/crag_val_40x.json"),  # crag 1-15
    "lizard_crag_test_16_multi": ("lizard_multiclass/img2021_40x", "lizard_multiclass/crag_test_40x.json"),  # crag 16-31
}

_PREDEFINED_SPLITS_PANNUKE_MULTI = {}
_PREDEFINED_SPLITS_PANNUKE_MULTI["pannuke_multiclass"] = {
    "pannuke_train_multi": ("PanNuke/Fold1/images/fold1/images", "PanNuke/Fold1/fold1.json"),
    "pannuke_eval_multi": ("PanNuke/Fold2/images/fold2/images", "PanNuke/Fold2/fold2.json"),
    "pannuke_test_multi": ("PanNuke/Fold3/images/fold3/images", "PanNuke/Fold3/fold3.json"),
}

_PREDEFINED_SPLITS_GLAND = {}
_PREDEFINED_SPLITS_GLAND["glas"] = {
    "glas_train": ("glas_dataset/image", "glas_dataset/glas_train.json"),  # 21-85
    "glas_val": ("glas_dataset/image", "glas_dataset/glas_val.json"),  # 1-20
    "glas_test": ("glas_dataset/image", "glas_dataset/glas_test.json"),
}

_PREDEFINED_SPLITS_GLAND_NUCLEUS = {}
_PREDEFINED_SPLITS_GLAND_NUCLEUS["glas_2class"] = {
    "glas_2class_train": ("glas_dataset/img2021_cropped", "glas_dataset/glas_2class_cropped_train.json"),
    "glas_2class_val": ("glas_dataset/img2021_cropped", "glas_dataset/glas_2class_cropped_test.json")
}

_PREDEFINED_SPLITS_NUCLEUS_PANOPTIC = {}
_PREDEFINED_SPLITS_NUCLEUS_PANOPTIC["kumar"] = {
    'kumar_train_panoptic': ("kumar/img_40x", "kumar/train_p.json", "kumar/train_panoptic", "kumar/train_panoptic.json", "kumar/panoptic_train"),
}

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


def register_all_nucleus(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_NUCLEUS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                {"thing_classes": ["nucleus"]},
                os.path.join(root, json_file),
                os.path.join(root, image_root),
            )
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_NUCLEUS_PANOPTIC.items():
        for key, (image_root, instances_json, panoptic_root, panoptic_json, semantic_root) in splits_per_dataset.items():
            register_coco_panoptic_separated(
                key,
                {"thing_classes": ["nucleus"]},
                os.path.join(root, image_root),
                os.path.join(root, panoptic_root),
                os.path.join(root, panoptic_json),
                os.path.join(root, semantic_root),
                os.path.join(root, instances_json),
            )

def register_all_gland(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_GLAND.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                {"thing_classes": ["gland"]},
                os.path.join(root, json_file),
                os.path.join(root, image_root),
            )

def register_all_gland_nucleus(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_GLAND_NUCLEUS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                {"thing_classes": ["nucleus", "gland"]},
                os.path.join(root, json_file),
                os.path.join(root, image_root),
            )

def register_all_nucleus_multi(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_NUCLEUS_MULTI.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                {"thing_classes": ["Neutrophil", "Epithelial", "Lymphocyte", "Plasma", "Neutrophil", "Connective"],
                 "thing_colors": [(241, 121, 45), (0, 176, 80), (255, 0, 0), (0, 191, 255), (241, 121, 45), (255, 255, 0)]},
                os.path.join(root, json_file),
                os.path.join(root, image_root),
            )
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_PANNUKE_MULTI.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                {"thing_classes": ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"],
                 "thing_colors": [(241, 121, 45), (0, 176, 80), (255, 0, 0), (0, 191, 255), (241, 121, 45)]},
                os.path.join(root, json_file),
                os.path.join(root, image_root),
            )
            


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets/detectron")
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
    register_all_nucleus(_root)
    register_all_gland(_root)
    register_all_gland_nucleus(_root)
    register_all_nucleus_multi(_root)
