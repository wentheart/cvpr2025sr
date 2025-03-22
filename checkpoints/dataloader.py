"""
这个文件简单进行数据预处理
"""
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import random

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=128, scale=4, is_train=True):
        """
        Args:
            lr_dir (str): 低分辨率图像目录
            hr_dir (str): 高分辨率图像目录
            crop_size (int): 训练时随机裁剪的大小
            scale (int): 超分辨率的放大倍数
            is_train (bool): 是否为训练模式
        """
        super(DIV2KDataset, self).__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.scale = scale
        self.is_train = is_train
        
        # 获取所有图像文件名
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))

        # 确保LR和HR图像对数量匹配
        assert len(self.lr_images) == len(self.hr_images), "LR和HR图像数量不匹配"
        
        # 基础变换
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        # 读取图像
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        if self.is_train:
            # 验证图像尺寸比例
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size
            
            assert hr_w == lr_w * self.scale and hr_h == lr_h * self.scale, \
                f"图像{idx}的LR和HR尺寸不符合scale比例"
            
            # 确保有足够的裁剪空间
            if lr_w < self.crop_size or lr_h < self.crop_size:
                # 如果图像太小，将其放大到可裁剪大小
                scale_factor = max(self.crop_size / lr_w, self.crop_size / lr_h)
                new_lr_w = int(lr_w * scale_factor)
                new_lr_h = int(lr_h * scale_factor)
                lr_img = lr_img.resize((new_lr_w, new_lr_h), Image.BICUBIC)
                hr_img = hr_img.resize((new_lr_w * self.scale, new_lr_h * self.scale), Image.BICUBIC)
                lr_w, lr_h = lr_img.size
            
            # 安全的随机裁剪
            max_lr_x = max(0, lr_w - self.crop_size)
            max_lr_y = max(0, lr_h - self.crop_size)
            x = random.randint(0, max_lr_x)
            y = random.randint(0, max_lr_y)
            
            # 执行裁剪
            lr_crop = lr_img.crop((x, y, x + self.crop_size, y + self.crop_size))
            
            # 对HR图像进行对应位置的裁剪
            hr_x = x * self.scale
            hr_y = y * self.scale
            hr_crop_size = self.crop_size * self.scale
            hr_crop = hr_img.crop((hr_x, hr_y, hr_x + hr_crop_size, hr_y + hr_crop_size))
            
            # 数据增强
            if random.random() < 0.5:
                lr_crop = lr_crop.transpose(Image.FLIP_LEFT_RIGHT)
                hr_crop = hr_crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                lr_crop = lr_crop.transpose(Image.FLIP_TOP_BOTTOM)
                hr_crop = hr_crop.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                lr_crop = lr_crop.rotate(angle)
                hr_crop = hr_crop.rotate(angle)
            
            # 转换为tensor
            lr_tensor = self.to_tensor(lr_crop)
            hr_tensor = self.to_tensor(hr_crop)
        else:
            # 验证模式不裁剪，只转换为tensor
            lr_tensor = self.to_tensor(lr_img)
            hr_tensor = self.to_tensor(hr_img)
        
        return lr_tensor, hr_tensor