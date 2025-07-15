from torch.utils.data import Dataset
from PIL import Image

class ColoredImageDataset(Dataset):
    def __init__(self, data_clean, data_noise, transforms=None):
        super(ColoredImageDataset, self).__init__()
        self.clean = data_clean
        self.noise = data_noise
        self.transforms = transforms

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        img = self.clean[idx]
        image_clean = Image.open(img).convert('L')
        noise = self.noise[idx]
        image_noise = Image.open(noise).convert('L')
        if self.transforms is not None:
            image_clean = self.transforms(image_clean)
            image_noise = self.transforms(image_noise)
        return image_noise, image_clean

