from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import os, cv2
from PIL import Image, ImageOps
from scipy import io
import random

def _get_file_list(data_path):
    """This function detects all image files within the specified parent
       directory for either training or testing. The path content cannot
       be empty, otherwise an error occurs.
    Args:
        data_path (str): Points to the directory where training or testing
                         data instances are stored.
    Returns:
        list, str: A sorted list that holds the paths to all file instances.
    """

    data_list = []

    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".mat")):
                    data_list.append(os.path.join(subdir, file))

    data_list.sort()

    if not data_list:
        raise FileNotFoundError("No data was found")

    return data_list
def _check_consistency(zipped_file_lists, n_total_files):
    """A consistency check that makes sure all files could successfully be
       found and stimuli names correspond to the ones of ground truth maps.
    Args:
        zipped_file_lists (tuple, str): A tuple of train and valid path names.
        n_total_files (int): The total number of files expected in the list.
    """

    assert len(list(zipped_file_lists)) == n_total_files, "Files are missing"

    for file_tuple in zipped_file_lists:
        file_names = [os.path.basename(entry) for entry in list(file_tuple)]
        file_names = [os.path.splitext(entry)[0] for entry in file_names]
        file_names = [entry.replace("_fixMap", "") for entry in file_names]
        file_names = [entry.replace("_fixPts", "") for entry in file_names]

        assert len(set(file_names)) == 1, "File name mismatch"


def _get_random_indices(list_length):
    """A helper function to generate an array of randomly shuffled indices
       to divide the MIT1003 and CAT2000 datasets into training and validation
       instances.
    Args:
        list_length (int): The number of indices that is randomly shuffled.
    Returns:
        array, int: A 1D array that contains the shuffled data indices.
    """

    indices = np.arange(list_length)
    prng = np.random.RandomState(42)
    prng.shuffle(indices)

    return indices

class SaliconDataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "SALICON/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")

        path = "train/"
        if not self.train:
            path = "val/"

        self.img_dir  = self.data_path + "stimuli/" + path
        self.gt_dir = self.data_path + "saliency/" + path
        self.fix_dir = self.data_path + "fixations/" + path

        self.img_ids = [nm.split(".")[0] for nm in os.listdir(self.img_dir)]

        self.exten = exten
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [384,384]
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            transforms.ColorJitter(
					brightness=0.4,
					contrast=0.4,
					saturation=0.4,
					hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.img_transform_val = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + ".mat")
        
        img = Image.open(img_path).convert('RGB')

        gt = Image.open(gt_path).convert('L')

        fixations = self.mat_loader(fix_path, (640, 480))
        fixations = self.pts2pil(fixations, img)

        if self.train:
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                gt = ImageOps.mirror(gt)
                fixations = ImageOps.mirror(fixations)

        gt = np.array(gt).astype('float')
        gt = cv2.resize(gt, (self.gt_size[0],self.gt_size[1]))

        fixations = np.array(fixations).astype('float')
        fixations = cv2.resize(fixations, (self.gt_size[0], self.gt_size[1]))

        if self.train:
            img = self.img_transform(img)
        else:
            img = self.img_transform_val(img)

        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')
        
        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)
    
    def __len__(self):
        return len(self.img_ids)

    def pts2pil(self, fixpts, img):
        fixmap = Image.new("L", img.size)
        for p in fixpts:
            fixmap.putpixel((p[0], p[1]), 255)
        return fixmap
    def mat_loader(self, path, shape):
        mat = io.loadmat(path)["gaze"]
        fix = []
        for row in mat:
            data = row[0].tolist()[2]
            for p in data:
                if p[0]<shape[0] and p[1]<shape[1]: # remove noise at the boundary.
                    fix.append(p.tolist())
        return fix


class ORSSDDataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "ORSSD-FP/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")

        path = "train/"
        if not self.train:
            path = "test/"

        self.img_dir = self.data_path + "stimuli/" + path
        self.gt_dir = self.data_path + "saliency/" + path
        self.fix_dir = self.data_path + "fixations/" + path

        self.img_ids = [nm.split(".")[0] for nm in os.listdir(self.img_dir)]

        self.exten = exten
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [384, 384]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.img_transform_val = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + ".mat")

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        fixations = Image.open(fix_path).convert('L')

        fixations = np.array(fixations).astype('float')
        fixations = cv2.resize(fixations, (self.gt_size[0], self.gt_size[1]))

        if self.train:
            img = self.img_transform_val(img)
        else:
            img = self.img_transform_val(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')

        assert np.min(gt) >= 0.0 and np.max(gt) <= 1.0
        assert np.min(fixations) == 0.0 and np.max(fixations) == 1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.img_ids)

def get_datasets(args):
    if args.dataset == 'SALICON':
        args.output_size = [384, 384]
        args.input_size = 384
        train_dataset = SaliconDataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                            input_size_w=args.input_size)
        val_dataset = SaliconDataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                          input_size_w=args.input_size)

    elif args.dataset == 'ORSSD':
        args.output_size = [384, 384]
        args.input_size = 384
        train_dataset = ORSSDDataset(args.dataset_dir, train=True, input_size_h=args.input_size,
                                            input_size_w=args.input_size)
        val_dataset = ORSSDDataset(args.dataset_dir, train=False, input_size_h=args.input_size,
                                          input_size_w=args.input_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)
    return train_loader, val_loader