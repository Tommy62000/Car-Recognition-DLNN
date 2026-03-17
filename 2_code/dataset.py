import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F
import scipy.io as sio
import numpy as np

class CompCarsDataset(Dataset):
    _cached_samples = None
    _cached_attributes = None
    _cached_make_names = None
    _cached_model_names = None

    # CONSTRUCTOR
    def __init__(self,
                 root_dir,
                 split='train',
                 split_ratio=0.8,
                 random_seed=42,
                 transform=None,
                 print_output=False):
        self.print_output = print_output

        assert split in ('train', 'test'), "split deve essere 'train' o 'test'"
        if self.print_output:
            print(f"{str.upper(split)}")

        # GENERATION OF PATHS
        self.root = root_dir
        self.img_dir = os.path.join(root_dir, 'image')
        self.label_dir = os.path.join(root_dir, 'label')
        self.attributes_file = os.path.join(root_dir, 'misc/attributes.txt')
        self.make_model_math_file = os.path.join(root_dir, 'misc/make_model_name.mat')
        if self.print_output:
            print(f"Established {split} dataset paths as:")
            print(f"Dataset root directory: {self.root}")
            print(f"Image directory: {self.img_dir}")
            print(f"Label directory: {self.label_dir}")
            print(f"Attributes file: {self.attributes_file}")
            print(f"Mathlab make and models file: {self.make_model_math_file}\n")

        # IMAGE PATH EXTRACTION
        if CompCarsDataset._cached_samples is None:
            all_samples = []
            if self.print_output:
                print(f"Starting {split} images extraction...", end=' ', flush=True)
            for dirpath, _, files in os.walk(self.img_dir):
                for fname in files:
                    if fname.lower().endswith('.jpg'):
                        full_path = os.path.join(dirpath, fname)
                        rel_path = os.path.relpath(full_path, self.img_dir).replace('\\', '/')
                        all_samples.append(rel_path)
            CompCarsDataset._cached_samples = all_samples
            if self.print_output:
                print("Done\n")
                print(f"Total number of image extracted: {len(all_samples)}")
        else:
            all_samples = CompCarsDataset._cached_samples

        # IMAGE SHUFFLING AND SPLITTING
        if self.print_output:
            print(f"Starting {split} images shuffling and selection of {split_ratio * 100}%...", end=' ', flush=True)
        random.seed(random_seed)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * split_ratio)
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        if self.print_output:
            print("Done\n")
            print(f"Total number if images selected: {len(self.samples)}")

        # ATTRIBUTES LOADING
        if CompCarsDataset._cached_attributes is None:
            self.attributes = {}
            if self.print_output:
                print(f"Starting attributes loading...", end=' ', flush=True)
            if self.attributes_file and os.path.exists(self.attributes_file):
                with open(self.attributes_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        # Skip header and empty lines
                        if not parts or parts[0].lower() == 'model_id':
                            continue
                        model_id = parts[0]
                        try:
                            self.attributes[model_id] = {
                                'max_speed': float(parts[1]),
                                'displacement': float(parts[2]),
                                'door_number': int(parts[3]),
                                'seat_number': int(parts[4]),
                                'type': int(parts[5])
                            }
                        except ValueError:
                            continue
                CompCarsDataset._cached_attributes = self.attributes
            else:
                if self.print_output:
                    print("Txt file not found, skipping")
        else:
            self.attributes = CompCarsDataset._cached_attributes
        if self.print_output:
            print(f"Done\n")
            print(f"Loaded attributes for {len(self.attributes)} models")

        # MAKE/MODEL LOADING
        if CompCarsDataset._cached_make_names is None or CompCarsDataset._cached_model_names is None:
            make_names, model_names = {}, {}
            if self.print_output:
                print(f"Starting make and models loading...", end=' ', flush=True)
            if self.make_model_math_file and os.path.exists(self.make_model_math_file):
                mat = sio.loadmat(self.make_model_math_file)
                makes = mat.get('make_names', np.array([])).squeeze()
                mods = mat.get('model_names', np.array([])).squeeze()
                for i, entry in enumerate(makes, 1):
                    make_names[str(i)] = str(entry)
                for i, entry in enumerate(mods, 1):
                    model_names[str(i)] = str(entry)
                CompCarsDataset._cached_make_names = make_names
                CompCarsDataset._cached_model_names = model_names
                if self.print_output:
                    print(f"Done\n")
                    print(f"Loaded {len(make_names)} makes names")
                    print(f"Loaded {len(model_names)} models names")
                try:
                    show = input("Print the make/model mapping? (y/n): ").strip().lower()
                except Exception:
                    show = 'n'
                if show == 'y':
                    print("Make names mapping:")
                    for k, v in make_names.items():
                        print(f"  {k}: {v}")
                    print("Model names mapping:")
                    for k, v in model_names.items():
                        print(f"  {k}: {v}")
            else:
                print("Mat file not found, skipping")
        else:
            self.make_names = CompCarsDataset._cached_make_names
            self.model_names = CompCarsDataset._cached_model_names

        # DATA TRANSFORM DEFINITION
        self.split = split
        self.resize_heigth = 600
        self.resize_width = 900
        self.transform = transform or T.Compose([
            T.Resize((self.resize_heigth, self.resize_width)),
            T.ToTensor()
        ])

    @staticmethod
    def get_resize_dims(transform):
        if isinstance(transform, T.Compose):
            for t in transform.transforms:
                if isinstance(t, T.Resize):
                    return t.size
        # Se è direttamente un Resize
        elif isinstance(transform, T.Resize):
            return transform.size
        return None
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # PATHS
        rel_path   = self.samples[idx]
        img_path   = os.path.join(self.img_dir, rel_path)
        label_path = os.path.join(self.label_dir, rel_path.replace('.jpg', '.txt'))

        # IMAGE TO TENSOR
        img = Image.open(img_path).convert('RGB')
        W_o, H_o = img.size
        img_tensor = self.transform(img)

        # BOUNDING BOX AND VIEWPOINT
        with open(label_path, 'r') as f:
            lines = f.read().splitlines()
        viewpoint = int(lines[0])
        x1_o, y1_o, x2_o, y2_o = map(int, lines[2].split())

        resize_dims = self.get_resize_dims(self.transform)
        if isinstance(resize_dims, tuple):
            resize_h, resize_w = resize_dims
        else:
            resize_h = resize_w = resize_dims

        sx = resize_w / W_o
        sy = resize_h / H_o

        x1 = int(x1_o * sx)
        y1 = int(y1_o * sy)
        x2 = int(x2_o * sx)
        y2 = int(y2_o * sy)
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.int)

        # SAMPLE COMPILATION
        make_id, model_id, year, _ = rel_path.split('/', 3)
        try:
            year = int(year)
        except ValueError:
            year = -1
        sample = {
            'image':     img_tensor,
            'viewpoint': viewpoint,
            'bbox':       bbox,
            'make_id':    int(make_id),
            'model_id':   int(model_id),
            'year':       int(year),
            'attributes': self.attributes.get(model_id, {})
        }
        return sample

def get_dataloaders(root,
                    train_ratio=0.8,
                    batch_size=16,
                    seed=42,
                    transform=None,
                    num_workers=4):
    
    train_ds = CompCarsDataset(root,
                               split='train',
                               split_ratio=train_ratio,
                               random_seed=seed,
                               transform=transform,
                               print_output=True)
    test_ds  = CompCarsDataset(root,
                               split='test',
                               split_ratio=1 - train_ratio,
                               random_seed=seed,
                               transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

