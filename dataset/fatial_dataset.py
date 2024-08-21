from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path
import csv

class FatialDataset(Dataset):
    """
    Face Dataset: Load images path and labels from csv file
    """
    def __init__(self, root_dir, transform = None, target_transform = None) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        root_dir = Path(root_dir)
        if root_dir.suffix == ".csv":
            self.root_dir = root_dir.parent
            csv_path = Path(root_dir)
        else:
            self.root_dir = root_dir
            csv_path = Path(root_dir, "labels.csv")
        self.images_path = []
        self.labels = []
        self._name = str(root_dir.name)
        with open(csv_path, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                self.images_path.append(Path(self.root_dir, row[1]))
                self.labels.append(row[2])
    
    @property
    def name(self):
        return self._name
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = read_image(str(image_path))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label