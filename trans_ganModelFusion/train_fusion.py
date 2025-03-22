# 标准库导入
import os
import sys
import time
import random
from pathlib import Path

# 第三方库导入
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# 设置环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 添加模型路径
MODULE_PATH = Path(__file__).parent.parent
sys.path.append(str(MODULE_PATH))

# 本地模型导入
from models.team00_DAT import DAT 
from models.team00_RFDN import RFDN
from Real_ESRGAN.RealESRGAN import RealESRGAN

class WeightPredictor(nn.Module):
    def __init__(self, bias=0.5):  # 添加偏置参数
        super(WeightPredictor, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.weight_net = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 3, 1)
        )
        
        # 使用偏置初始化
        self.bias = bias
        
    def forward(self, x):
        feat = self.feature_net(x)
        weights = self.weight_net(feat)
        weights = weights.squeeze(-1).squeeze(-1)  # [B, 3]
        
        # 添加偏置并使用softmax确保和为1
        weights[:, 0] = weights[:, 0] + self.bias  # ��RealESRGAN添加正偏置
        weights = F.softmax(weights, dim=1)
        return weights

class FusionModel(nn.Module):
    def __init__(self, device='cuda'):
        super(FusionModel, self).__init__()
        self.device = device
        # 初始化模型并移至指定设备
        self.realesrgan = RealESRGAN(device, scale=4)
        self.realesrgan.load_weights("./model_zoo/RealESRGAN_x4plus.pth", download=False)
        
        self.dat = DAT(upscale=4).to(device)
        self.dat.load_state_dict(torch.load("./model_zoo/team00_dat.pth"))
        
        self.rfdn = RFDN(upscale=4).to(device)
        self.rfdn.load_state_dict(torch.load("./model_zoo/team00_rfdn.pth"))
        
        self.predictor = WeightPredictor().to(device)
        
        # 设置评估模式
        self.dat.eval()
        self.rfdn.eval()
        
    def forward(self, x):
        batch_size = x.size(0)
        with torch.no_grad():
            # 处理RealESRGAN输入
            out1 = []
            for i in range(batch_size):
                x_single = x[i]
                x_pil = transforms.ToPILImage()(x_single.cpu())  # CPU上进行PIL转换
                sr_img = self.realesrgan.predict(x_pil)
                sr_tensor = transforms.ToTensor()(sr_img).to(self.device)  # 转回GPU
                out1.append(sr_tensor)
            out1 = torch.stack(out1).to(self.device)  # 确保在GPU上
            
            # DAT和RFDN直接处理
            out2 = self.dat(x)  
            out3 = self.rfdn(x)
        
        # 预测权重
        weights = self.predictor(x)  # [B, 3]
        weights = weights.view(batch_size, 3, 1, 1, 1)
        
        # 加权融合
        out = (weights[:,0] * out1 + 
               weights[:,1] * out2 + 
               weights[:,2] * out3)
        
        return out

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=128, scale=4, is_train=True):
        super(DIV2KDataset, self).__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.scale = scale
        self.is_train = is_train
        
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        assert len(self.lr_images) == len(self.hr_images)
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        if self.is_train:
            # 验证图像尺寸比例
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size
            assert hr_w == lr_w * self.scale and hr_h == lr_h * self.scale
            
            # 确保有足够的裁剪空间
            if lr_w < self.crop_size or lr_h < self.crop_size:
                raise ValueError(f"图像{idx}小于裁剪尺寸")
            
            # 随机裁剪
            max_lr_x = max(0, lr_w - self.crop_size)
            max_lr_y = max(0, lr_h - self.crop_size)
            x = random.randint(0, max_lr_x)
            y = random.randint(0, max_lr_y)
            
            lr_crop = lr_img.crop((x, y, x + self.crop_size, y + self.crop_size))
            hr_x = x * self.scale
            hr_y = y * self.scale
            hr_crop_size = self.crop_size * self.scale
            hr_crop = hr_img.crop((hr_x, hr_y, hr_x + hr_crop_size, hr_y + hr_crop_size))
            
            # 数据增强
            if random.random() < 0.5:
                lr_crop = lr_crop.transpose(Image.FLIP_LEFT_RIGHT)
                hr_crop = hr_crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                lr_crop = lr_crop.transpose(Image.FLIP_TOP_BOTTOM)
                hr_crop = hr_crop.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                lr_crop = lr_crop.rotate(angle)
                hr_crop = hr_crop.rotate(angle)
                
            return self.to_tensor(lr_crop), self.to_tensor(hr_crop)
        else:
            return self.to_tensor(lr_img), self.to_tensor(hr_img)

def get_random_indices(total_size, sample_size=32):
    """随机抽取指定数量的索引"""
    indices = torch.randperm(total_size)[:sample_size]
    return indices

class BiasedFusionLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_psnr=1.0, lambda_reg=0.1, target_weight=0.5):
        super(BiasedFusionLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_psnr = lambda_psnr
        self.lambda_reg = lambda_reg
        self.target_weight = target_weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, sr, hr, weights):
        # 基础重建损失
        l1_loss = self.l1_loss(sr, hr)
        
        # PSNR损失
        mse_loss = nn.MSELoss()(sr, hr)
        psnr_loss = mse_loss
        
        # 偏好损失 - 鼓励RealESRGAN权重接近目标值
        preference_loss = F.smooth_l1_loss(weights[:, 0], 
                                         torch.ones_like(weights[:, 0]) * self.target_weight)
        
        # 总损失
        total_loss = (self.lambda_l1 * l1_loss + 
                     self.lambda_psnr * psnr_loss + 
                     self.lambda_reg * preference_loss)
        
        return total_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionModel(device).to(device)
    
    train_dataset = DIV2KDataset(
        lr_dir='./data/DIV2K_train_LR',
        hr_dir='./data/DIV2K_train_HR', 
        is_train=True
    )
    valid_dataset = DIV2KDataset(
        lr_dir='./data/DIV2K_valid_LR',
        hr_dir='./data/DIV2K_valid_HR',
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=80,
        shuffle=True,
        num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False
    )
    
    criterion = BiasedFusionLoss(
        lambda_l1=1.0,
        lambda_psnr=1.0,
        lambda_reg=0.2,
        target_weight=0.5  # 期望RealESRGAN的权重
    )
    optimizer = optim.Adam(model.predictor.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    best_psnr = 0
    for epoch in range(100):  # 修改为100轮
        model.train()
        for lr, hr in tqdm(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr, model.predictor(lr))  # 添加权重参数
            loss.backward()
            optimizer.step()
            
        model.eval()
        psnr_list = []
        with torch.no_grad():
            # 获取验证集总数
            total_samples = len(valid_loader.dataset)
            # 随机抽取16张图片的索引
            sampled_indices = get_random_indices(total_samples, sample_size=16)
            
            for idx, (lr, hr) in enumerate(valid_loader):
                # 只处理被抽样的图片
                if idx in sampled_indices:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    mse = nn.MSELoss()(sr, hr)
                    psnr = -10 * torch.log10(mse)
                    psnr_list.append(psnr.item())
                    
                    # 如果已经处理完所有抽样图片，就退出循环
                    if len(psnr_list) >= 16:
                        break
                
        avg_psnr = np.mean(psnr_list)
        
        # 创建保存目录
        os.makedirs('./checkpoints/best_pth', exist_ok=True)
        os.makedirs('./checkpoints', exist_ok=True)
        
        # 保存最优模型
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(
                model.state_dict(), 
                './checkpoints/best_pth/best_fusion.pth'
            )
        
        # 每20个epoch保存一次检查点
        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                f'./checkpoints/fusion_epoch_{epoch+1}.pth'
            )
            
        scheduler.step()
        print(f'Epoch {epoch+1+60}/100, PSNR: {avg_psnr:.2f}')

if __name__ == '__main__':
    train()