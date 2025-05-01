import csv
import os
import random
import torch
import re
from collections import defaultdict
from itertools import product
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw
from tqdm import tqdm
import colorsys
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader


def hsv_to_rgb(h, s, v):
    """Convert HSV (0-360, 0-100, 0-100) to RGB (0-255, 0-255, 0-255)."""
    r, g, b = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)
    return (int(r * 255), int(g * 255), int(b * 255))


def generate_shapes(output_path: str, num_shapes: int = 10, seed: int = 0):
    random.seed(seed)

    # Base hues
    base_hues = {
        'red': 0,    # 0 degrees hue
        'blue': 240  # 240 degrees hue
    }
    image_size = (512, 512)
    hue_variation = 30  # Degrees
    sat = 90            # 90% saturation
    val = 90            # 90% value (brightness)

    labels = []

    for i in tqdm(range(num_shapes)):
        # Create a blank white canvas
        img = Image.new('RGB', image_size, color='black')
        draw = ImageDraw.Draw(img)

        # Randomly choose shape, color, size
        shape = random.choice(['circle', 'triangle'])
        color_name = random.choice(['red', 'blue'])
        base_hue = base_hues[color_name]

        # Slight hue variation (up to +/- hue_variation degrees)
        hue = (base_hue + random.randint(-hue_variation, hue_variation)) % 360

        # Convert HSV to RGB
        color = hsv_to_rgb(hue, sat, val)

        # Size variation
        size = random.choice(['small', 'large'])
        if size == 'small':
            radius = random.randint(150, 200)
        else:
            radius = random.randint(400, 450)

        # Random location
        x = (image_size[0] - radius) // 2
        y = (image_size[1] - radius) // 2

        if shape == 'circle':
            draw.ellipse((x, y, x + radius, y + radius), fill=color, outline=None)
        elif shape == 'triangle':
            draw.polygon([
                (x + radius // 2, y),
                (x, y + radius),
                (x + radius, y + radius)
            ], fill=color, outline=None)

        img.save(os.path.join(output_path, f"{i}.png"))

        color_label = int(color_name == 'red')
        shape_label = int(shape == 'circle')
        size_label = int(size == 'large')
        label = [color_label, shape_label, size_label]
        labels.append(label)

    with open(os.path.join(output_path, "labels.csv"), "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(["idx", "color", "shape", "size", "tuple"])
        for j, row in enumerate(labels):
            csv_out.writerow([j] + row + [tuple(row)])


# class Shapes(Dataset):
# 
#     def __init__(
#         self,
#         path: str,
#         num_samples: int,
#         labels: Optional[List[Tuple[int, int, int]]] = None,
#         transforms: transforms.Compose = None,
#     ):
#         self.path = path
#         self.labels = labels
#         self.transforms = transforms
# 
#         labels_csv = os.path.join(self.path, "labels.csv")
#         self.all_labels = pd.read_csv(labels_csv, index_col="idx")
# 
#         if self.labels is not None:
#             mask = self.all_labels.tuple.isin([str(l) for l in self.labels])
#             self.all_labels = self.all_labels[mask]
#         
#         self.all_labels = self.all_labels.reset_index().iloc[:num_samples]
# 
#     def __getitem__(self, idx):
#         row = self.all_labels.iloc[idx]
#         img_path = os.path.join(self.path, f"{row.idx}.png")
#         img = Image.open(img_path)
#         if self.transforms is not None:
#             img = self.transforms(img)
#         return img, [row["color"], row["shape"], row["size"]]
# 
#     def __len__(self):
#         return len(self.all_labels)


class Shapes(Dataset):

    def __init__(
        self,
        path: str,
        num_samples: int,
        labels: List[Tuple[int, int, int]],
        transforms: transforms.Compose = None,
        test: bool = False,
    ):
        path = os.path.join(path, "test" if test else "train")
        self.transforms = transforms

        label_strs = ["".join([str(l) for l in label]) for label in labels]
        self.paths = []
        counter = defaultdict(int)
        for file in os.listdir(path):  # , label in self.all_paths:
            label = re.search(r"CLEVR_(\d{3})_\d+\.png", file)
            if not label:
                continue
            label = label.group(1) 
            if label in label_strs and counter[label] < num_samples:
                self.paths.append((os.path.join(path, file), labels[label_strs.index(label)]))
                counter[label] += 1
        
    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img = Image.open(path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img[:3], label

    def __len__(self):
        return len(self.paths)


def preprocess(pixel_size: int):
    return transforms.Compose([
        transforms.Resize((pixel_size, pixel_size)),
        transforms.ToTensor(),
    ])


def aggregrate_datasets(path: str, pixel_size: int = 28, train_subset_size: int = 1000, test_subset_size: int = 50):
    tfs = preprocess(pixel_size)

    train_dicts = {}
    test_dicts = {}
    labels = list(product(*[(0, 1), (0, 1), (0, 1)]))
    for label in labels:
        train_ds = Shapes(path, train_subset_size, [label], tfs, False)
        test_ds = Shapes(path, test_subset_size, [label], tfs, True)
        train_dicts[label] = train_ds
        test_dicts[label] = test_ds

    return train_dicts, test_dicts


def diffusion_ds(path: str, train_labels: List[Tuple[int, int, int]], pixel_size: int = 32):
    train_dicts, test_dicts = aggregrate_datasets(path, pixel_size, 1000, 50)
    train_ds = ConcatDataset([train_dicts[l] for l in train_labels])
    return train_ds, test_dicts


def classifier_ds(path: str, pixel_size: int = 28):
    train_dicts, test_dicts = aggregrate_datasets(path, pixel_size, 1000, 1000)
    train_ds = ConcatDataset(list(train_dicts.values()))
    test_ds = ConcatDataset(list(test_dicts.values()))
    return train_ds, test_ds


if __name__ == "__main__":
    seed = 0
    path = "./datasets/shapes"
    num_shapes = 20000

    os.makedirs(path, exist_ok=True)
    generate_shapes(path, num_shapes, seed)

    # train_labels = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0)]
    # train_ds, test_ds = classifier_ds(path)
    # for batch in tqdm(DataLoader(train_ds)):
    #     pass
