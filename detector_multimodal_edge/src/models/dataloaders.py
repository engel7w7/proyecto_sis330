import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.config import BATCH_SIZE, IMG_SIZE

# Transformaciones profesionales de ImageNet
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultimodalDeepfakeDataset(Dataset):
    """
    Dataset Genérico MLOps para imágenes y espectrogramas.
    Estructura de carpetas esperada:
      data_dir/
        ├── real/
        └── fake/
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')

        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(real_dir, fname), 0.0))

        if os.path.exists(fake_dir):
            for fname in os.listdir(fake_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(fake_dir, fname), 1.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

def obtener_dataloaders(data_dir, batch_size=BATCH_SIZE):
    dataset = MultimodalDeepfakeDataset(data_dir, transform=transform_train)
    if len(dataset) == 0:
        return None, None

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
