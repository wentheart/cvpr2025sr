"""
融合网络改进版本，加入验证集和PSNR指标
"""
import os
import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import time
import math

from models.team00_DAT import DAT
from models.team00_RFDN import RFDN
from models.team00_SwinIR import SwinIR
from dataloader import DIV2KDataset
import smtplib
from email.mime.text import MIMEText
from email.header import Header

class AttentionWeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Softmax(dim=1)  # 使用Softmax确保权重和为1
        )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        weights = self.attention(feat)
        return weights

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.alpha * l1 + self.beta * mse

def get_random_indices(total_size, sample_size=32):
    """随机抽取指定数量的索引"""
    indices = torch.randperm(total_size)[:sample_size]
    return indices.tolist()

def main():
    # 创建保存目录
    save_dir = './checkpoints/base_fusion_result'
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    print("正在加载基础模型...")
    dat_model = DAT().to(device).eval()
    swinir_model = SwinIR().to(device).eval()
    rfdn_model = RFDN().to(device).eval()

    # 加载预训练权重
    dat_model.load_state_dict(torch.load("./model_zoo/team00_dat.pth"))
    swinir_model.load_state_dict(torch.load("./model_zoo/team00_swinir.pth"))
    rfdn_model.load_state_dict(torch.load("./model_zoo/team00_rfdn.pth"))

    # 初始化权重网络和优化器
    weight_net = AttentionWeightNet().to(device)
    weight_net.load_state_dict(torch.load("./checkpoints/base_fusion_result/fusion_weights_epoch100.pth")['weight_net_state_dict'])

    optimizer = optim.Adam(weight_net.parameters(), lr=1e-4)
    optimizer.load_state_dict(torch.load("./checkpoints/base_fusion_result/fusion_weights_epoch100.pth")['optimizer_state_dict'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scheduler.load_state_dict(torch.load("./checkpoints/base_fusion_result/fusion_weights_epoch100.pth")['scheduler_state_dict'])

    criterion = CombinedLoss()

    # 数据加载
    print("正在加载数据集...")
    train_dataset = DIV2KDataset(
        lr_dir='./data/DIV2K_train_LR',
        hr_dir='./data/DIV2K_train_HR',
        is_train=True
    )
    val_dataset = DIV2KDataset(
        lr_dir='./data/DIV2K_valid_LR',
        hr_dir='./data/DIV2K_valid_HR',
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=72, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 训练循环
    print("开始训练...")
    num_epochs = 100
    best_psnr = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        weight_net.train()
        epoch_loss = 0
        start_time = time.time()
        
        for i, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            # 获取三个模型的输出
            with torch.no_grad():
                out_dat = dat_model(lr)
                out_swinir = swinir_model(lr)
                out_rfdn = rfdn_model(lr)
            
            # 拼接输出并预测权重
            combined = torch.cat([out_dat, out_swinir, out_rfdn], dim=1)
            weights = weight_net(combined)
            
            # 加权融合
            w1, w2, w3 = weights.chunk(3, dim=1)
            fused = w1 * out_dat + w2 * out_swinir + w3 * out_rfdn
            
            # 计算损失
            loss = criterion(fused, hr)
            epoch_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # 验证阶段
        weight_net.eval()
        val_psnr = 0
        
        # 获取验证集总数并随机抽样
        total_val_samples = len(val_loader.dataset)
        sampled_indices = get_random_indices(total_val_samples, sample_size=28)  # 随机抽取28张图片
        
        with torch.no_grad():
            for idx, (lr, hr) in enumerate(val_loader):
                # 只处理被抽样的图片
                if idx not in sampled_indices:
                    continue
                    
                lr, hr = lr.to(device), hr.to(device)
                
                out_dat = dat_model(lr)
                out_swinir = swinir_model(lr)
                out_rfdn = rfdn_model(lr)
                
                combined = torch.cat([out_dat, out_swinir, out_rfdn], dim=1)
                weights = weight_net(combined)
                
                w1, w2, w3 = weights.chunk(3, dim=1)
                fused = w1 * out_dat + w2 * out_swinir + w3 * out_rfdn
                
                val_psnr += calculate_psnr(fused, hr)
        
        # 使用实际验证的图片数量计算平均PSNR
        avg_val_psnr = val_psnr / len(sampled_indices)
        scheduler.step(avg_val_psnr)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Val PSNR: {avg_val_psnr:.2f}, Time: {time.time()-start_time:.2f}s")

        # 每25个epoch保存一次模型
        if (epoch + 1) % 25 == 0:
            save_path = os.path.join(save_dir, f'fusion_weights_epoch{epoch+1+100}.pth')
            torch.save({
                'epoch': epoch + 1,
                'weight_net_state_dict': weight_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'psnr': avg_val_psnr,
            }, save_path)
            print(f"模型已保存到: {save_path}")
        
        # 保存最佳模型
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            best_save_path = os.path.join(save_dir, 'fusion_weights_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'weight_net_state_dict': weight_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'psnr': best_psnr,
            }, best_save_path)
            print(f"最佳模型已保存到: {best_save_path}，PSNR: {best_psnr:.2f}")

if __name__ == '__main__':
    main()

