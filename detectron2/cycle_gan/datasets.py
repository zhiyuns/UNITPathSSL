import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import detectron2.data.detection_utils as utils


class ImageDataset(Dataset):
    def __init__(self, root, root_B, target='nucleus_and_gland', scale=None, img_format='RGB', seg_target='ISG'):
        self.name = 'CycleDataset'
        self.target = target
        self.scale = scale  # have a format of [scale_pix, ] (for example, [40x_256, ])
        self.img_format = img_format
        self.seg_target = seg_target

        self.files_A = []
        # may have multiple image root, each have format /your/mask/image/path(-num_of_image)
        for each_root_a in root:
            if len(each_root_a.split('-')) == 1:
                # if image path end with path
                each_files = sorted(glob.glob(os.path.join(each_root_a, "*.png")))
                random.shuffle(each_files)
                self.files_A.extend(each_files)
            else:
                # if image path end with path(-num_of_image)
                num_img_a = int(each_root_a.split('/')[-2].split('-')[1])
                each_root_a = each_root_a.replace(f'-{num_img_a}', '')
                each_files = os.listdir(each_root_a)
                each_files = sorted([os.path.join(each_root_a, x) for x in each_files])
                random.shuffle(each_files)
                self.files_A.extend(each_files[:num_img_a])
        random.shuffle(self.files_A)
        self.files_B = []
        # may have multiple image root, each have format /your/pathology/image/path(-num_of_image)
        for each_root_b in root_B:
            if len(each_root_b.split('-')) == 1:
                # if image path end with path
                each_files = os.listdir(each_root_b)
                each_files = sorted([os.path.join(each_root_b, x) for x in each_files])
                random.shuffle(each_files)
                self.files_B.extend(each_files)
            else:
                # if image path end with path(-num_of_image)
                num_img_b = int(each_root_b.split('/')[-2].split('-')[1])
                each_root_b = each_root_b.replace(f'-{num_img_b}', '')
                each_files = sorted(glob.glob(os.path.join(each_root_b, "*.png")))
                random.shuffle(each_files)
                self.files_B.extend(each_files[:num_img_b])
        random.shuffle(self.files_B)
        print(f'dataset A length: {len(self.files_A)}, dataset B length: {len(self.files_B)},')

        raw_shape = [len(self.files_A)] + list(self._load_raw_image(0).shape)
        self._raw_shape = list(raw_shape)
        raw_labels = self._get_raw_labels()
        self._label_shape = raw_labels.shape[1:]

    def __getitem__(self, index):
        record = {}
        img_name_B = self.files_B[index % len(self.files_B)]
        img_name = self.files_A[index % len(self.files_A)]
        dir_name = os.path.dirname(img_name)
        image_A = self.get_desired_img_format(cv2.imread(img_name))
        image_B = self.get_desired_img_format(cv2.imread(img_name_B))

        scale, target_size = random.choice(self.scale).split('_')
        target_size = int(target_size)
        magnitude = 40 // int(scale.split('x')[0])
        ori_size = int(target_size * magnitude)
        # x denotes to width, y denotes to height
        x, y = random.randint(0, image_A.shape[1] - ori_size), random.randint(0, image_A.shape[0] - ori_size)
        box_a = (x, y, x + ori_size, y + ori_size)
        size_b = int(ori_size * (random.random() * 0.2 + 0.8))  # in range (0.8 1)
        x, y = random.randint(0, image_B.shape[1] - size_b), random.randint(0, image_B.shape[0] - size_b)
        box_b = (x, y, x + size_b, y + size_b)
        image_A = image_A[box_a[1]:box_a[3], box_a[0]:box_a[2], :]
        image_B = image_B[box_b[1]:box_b[3], box_b[0]:box_b[2], :]
        image_A = cv2.resize(image_A, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        image_B = cv2.resize(image_B, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        position_idx = img_name.split('/')[-1].split('.')[0].split('_')
        if len(position_idx) == 3:
            x_idx = int(position_idx[1])
            y_idx = int(position_idx[2])
            box = (x_idx * target_size, y_idx * target_size, (x_idx + 1) * target_size, (y_idx + 1) * target_size)
        else:
            box = box_a

        target = target_gland = None
        if 'nucleus' in self.target:
            npy_file_name = os.path.join(dir_name.replace(dir_name.split('/')[-1], 'cell_mask'),
                                         img_name.split('/')[-1].split('.')[0].split('_')[0] + '.npy')
            target = utils.get_target(file_path=npy_file_name, img_index=index,
                                           box=box, target_size=(target_size, target_size), mode=self.seg_target)
        if 'gland' in self.target:
            gland_mask_file_name = os.path.join(dir_name.replace(dir_name.split('/')[-1], 'gland_img'),
                                                img_name.split('/')[-1].split('.')[0].split('_')[0] + '.png')
            target_gland = utils.get_target(file_path=gland_mask_file_name, img_index=index,
                                                 box=box, target_size=(target_size, target_size), category=1)
        if self.seg_target == 'ISG':
            while target == 0 or target_gland == 0:
                del self.files_A[index % len(self.files_A)]
                target = target_gland = None
                img_name = self.files_A[index % len(self.files_A)]
                position_idx = img_name.split('/')[-1].split('.')[0].split('_')
                if len(position_idx) == 3:
                    x_idx = int(position_idx[1])
                    y_idx = int(position_idx[2])
                    box = (x_idx * target_size, y_idx * target_size, (x_idx + 1) * target_size, (y_idx + 1) * target_size)
                else:
                    box = box_a
                dir_name = os.path.dirname(img_name)
                if 'nucleus' in self.target:
                    npy_file_name = os.path.join(dir_name.replace(dir_name.split('/')[-1], 'cell_mask'),
                                                 img_name.split('/')[-1].split('.')[0].split('_')[0] + '.npy')
                    target = utils.get_target(file_path=npy_file_name, img_index=index,
                                                   box=box, target_size=(target_size, target_size), mode=self.seg_target)
                if 'gland' in self.target:
                    gland_mask_file_name = os.path.join(dir_name.replace(dir_name.split('/')[-1], 'gland_img'),
                                                        img_name.split('/')[-1].split('.')[0].split('_')[0] + '.png')
                    target_gland = utils.get_target(file_path=gland_mask_file_name, img_index=index,
                                                         box=box, target_size=(target_size, target_size), category=1)

        if 'gland' in self.target:
            target_gland.extend(target)
            target = target_gland
        record["file_name"] = img_name
        record["file_name_B"] = img_name_B
        record["box_a"] = box
        record["image_a"] = image_A
        record["image_b"] = image_B
        record["image_id"] = index
        record["height"] = record["width"] = target_size
        if self.seg_target == 'ISG':
            record["annotations"] = target
        elif self.seg_target == 'SSG':
            record["sem_seg"] = target
        record["c_label"] = self.get_label(index)

        return record

    def get_desired_img_format(self, image):
        """
            Convert image numpy array to numpy array of target format.

            Args:
                image (np.ndarray): a numpy format image read by opencv

            Returns:
                (np.ndarray): also see `read_image`
        """
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if self.img_format == "L":
            image = image[:, :, ::-1]
            image = np.expand_dims(image, -1)
        elif self.img_format == "RGB":
            # flip channels if needed
            image = image[:, :, ::-1]
        elif self.img_format == "YUV-BT.601":
            image = image[:, :, ::-1]
            image = image / 255.0
            _M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
            image = np.dot(image, np.array(_M_RGB2YUV).T)

        return image

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __call__(self):
        return self

    def _load_raw_image(self, raw_idx):
        size = int(self.scale[0].split('_')[1])
        image = self.get_desired_img_format(cv2.imread(self.files_B[raw_idx % len(self.files_B)]))
        image = image[:size, :size, :]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _get_raw_labels(self):
        return np.zeros([self._raw_shape[0], 0], dtype=np.float32)

    @property
    def label_shape(self):
        return list(self._label_shape)

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    def get_label(self, idx):
        # label = self._get_raw_labels()[idx]
        # return label.copy()
        return np.zeros([0], dtype=np.float32)  # do not return any label, may change in the future

    def get_cond_img(self, idx):
        random.seed(idx)
        scale, target_size = random.choice(self.scale).split('_')
        target_size = int(target_size)
        magnitude = 40 // int(scale.split('x')[0])
        ori_size = int(target_size * magnitude)
        img_name = self.files_A[idx % len(self.files_A)]
        image_A = self.get_desired_img_format(cv2.imread(img_name))
        x, y = random.randint(0, image_A.shape[1] - ori_size), random.randint(0, image_A.shape[0] - ori_size)
        box_a = (x, y, x + ori_size, y + ori_size)
        image_A = image_A[box_a[1]:box_a[3], box_a[0]:box_a[2], :]
        image_A = cv2.resize(image_A, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        # image_A = torch.as_tensor(np.ascontiguousarray(image_A.transpose(2, 0, 1)))
        return image_A

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]
