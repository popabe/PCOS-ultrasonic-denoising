import os
import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ColoredImageDataset
from diffusion_model import DiffusionModel

# 项目参数
latents_size = 256
layers_per_block = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 300
batch_size = 40
IMG_SIZE = 256

# 数据文件路径
data_clean = sorted(glob.glob('data/clean/*.png'))
data_noise = sorted(glob.glob('data/noise/*.png'))

# 数据增强
transforms_ = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# 训练集与测试集
train_dl = DataLoader(
    ColoredImageDataset(data_clean, data_noise, transforms=transforms_),
    batch_size=batch_size, shuffle=True, num_workers=0
)
test_dl = DataLoader(
    ColoredImageDataset(data_clean, data_noise, transforms=transforms_),
    batch_size=1, shuffle=True, num_workers=0
)

# 初始化模型
diffusion_model = DiffusionModel(latents_size=latents_size, layers_per_block=layers_per_block, device=device)
diffusion_model.initialize_opt_loss_function()

# 训练
diffusion_model.train_model(
    dataset=train_dl, 
    test_dl=test_dl, 
    epochs=epochs, 
    verbose=50, 
    device=device, 
    batch_size=batch_size, 
    latent_size=latents_size, 
    starting_epoch=1
)
