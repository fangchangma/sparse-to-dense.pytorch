import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import transforms

IMG_EXTENSIONS = [
    '.h5',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        # print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])

    return rgb, depth

iheight, iwidth = 480, 640 # raw image size
oheight, owidth = 228, 304 # image size after pre-processing
color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

def train_transform(rgb, depth):
    s = np.random.uniform(1.0, 1.5) # random scaling
    # print("scale factor s={}".format(s))
    depth_np = depth / s
    angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

    # perform 1st part of data augmentation
    transform = transforms.Compose([
        transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation is very slow
        transforms.Rotate(angle),
        transforms.Resize(s),
        transforms.CenterCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    rgb_np = transform(rgb)

    # random color jittering 
    rgb_np = color_jitter(rgb_np)

    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np

def val_transform(rgb, depth):
    depth_np = depth

    # perform 1st part of data augmentation
    transform = transforms.Compose([
        transforms.Resize(240.0 / iheight),
        transforms.CenterCrop((oheight, owidth)),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np

def rgb2grayscale(rgb):
    return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114


to_tensor = transforms.ToTensor()

class NYUDataset(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'

    def __init__(self, root, type, modality='rgb', num_samples=0, loader=h5_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if type == 'train':
            self.transform = train_transform
        elif type == 'val':
            self.transform = val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader

        if modality in self.modality_names:
            self.modality = modality
            if modality in ['rgbd', 'd', 'gd']:
                if num_samples <= 0:
                    raise (RuntimeError("Invalid number of samples: {}\n".format(num_samples)))
                self.num_samples = num_samples
            else:
                self.num_samples = 0
        else:
            raise (RuntimeError("Invalid modality type: " + modality + "\n"
                                "Supported dataset types are: " + ''.join(self.modality_names)))

    def create_sparse_depth(self, depth, num_samples):
        prob = float(num_samples) / depth.size
        mask_keep = np.random.uniform(0, 1, depth.shape) < prob
        sparse_depth = np.zeros(depth.shape)
        sparse_depth[mask_keep] = depth[mask_keep]
        return sparse_depth

    def create_rgbd(self, rgb, depth, num_samples):
        sparse_depth = self.create_sparse_depth(depth, num_samples)
        # rgbd = np.dstack((rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], sparse_depth))
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth

    def __get_all_item__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor, input_np, depth_np) 
        """
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)
        
        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np, self.num_samples)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(depth_np, self.num_samples)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor, input_np, depth_np

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor) 
        """
        input_tensor, depth_tensor, input_np, depth_np = self.__get_all_item__(index)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)
