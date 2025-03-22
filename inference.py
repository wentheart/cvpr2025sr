import os
import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from baseModelFusionNet import AttentionWeightNet

import sys
from models.team00_DAT import DAT
from models.team00_RFDN import RFDN
from models.team00_SwinIR import SwinIR

class SelfEnsembleStrategy(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        
    def forward(self, x):
        # 1. 准备所有变换后的输入
        inputs = []
        # 原始输入
        inputs.append(x)
        # 水平翻转
        inputs.append(torch.flip(x, dims=[-1]))
        # 垂直翻转
        inputs.append(torch.flip(x, dims=[-2]))
        # 旋转90度
        inputs.append(torch.rot90(x, k=1, dims=[-2,-1]))
        # 旋转180度
        inputs.append(torch.rot90(x, k=2, dims=[-2,-1]))
        # 旋转270度
        inputs.append(torch.rot90(x, k=3, dims=[-2,-1]))
        # 水平翻转+旋转90度
        inputs.append(torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2,-1]))
        # 垂直翻转+旋转90度
        inputs.append(torch.rot90(torch.flip(x, dims=[-2]), k=1, dims=[-2,-1]))
        
        # 2. 批量处理所有输入
        outputs = []
        for x_trans in inputs:
            outputs.append(self.model(x_trans))
            
        # 3. 对预测结果进行相应的逆变换
        restored = []
        # 原始输出无需变换
        restored.append(outputs[0])
        # 水平翻转的逆变换
        restored.append(torch.flip(outputs[1], dims=[-1]))
        # 垂直翻转的逆变换
        restored.append(torch.flip(outputs[2], dims=[-2]))
        # 旋转90度的逆变换
        restored.append(torch.rot90(outputs[3], k=-1, dims=[-2,-1]))
        # 旋转180度的逆变换
        restored.append(torch.rot90(outputs[4], k=-2, dims=[-2,-1]))
        # 旋转270度的逆变换
        restored.append(torch.rot90(outputs[5], k=-3, dims=[-2,-1]))
        # 水平翻转+旋转90度的逆变换
        restored.append(torch.flip(torch.rot90(outputs[6], k=-1, dims=[-2,-1]), dims=[-1]))
        # 垂直翻转+旋转90度的逆变换
        restored.append(torch.flip(torch.rot90(outputs[7], k=-1, dims=[-2,-1]), dims=[-2]))
        
        # 4. 融合所有结果
        return torch.mean(torch.stack(restored), dim=0)

def inference(mode):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    save_dir = f'./baseModelFusion/results/{mode}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载并初始化模型
    print("正在加载模型...")
    dat_model = DAT().to(device).eval()
    swinir_model = SwinIR().to(device).eval()
    rfdn_model = RFDN().to(device).eval()
    weight_net = AttentionWeightNet().to(device).eval()
    
    # 加载预训练权重
    dat_model.load_state_dict(torch.load("./model_zoo/team00_dat.pth"))
    swinir_model.load_state_dict(torch.load("./model_zoo/team00_swinir.pth"))
    rfdn_model.load_state_dict(torch.load("./model_zoo/team00_rfdn.pth"))
    checkpoint = torch.load('./checkpoints/fusion_weights_best.pth')
    weight_net.load_state_dict(checkpoint['weight_net_state_dict'])
    
    # 设置图像转换
    transform = transforms.ToTensor()
    
    # 获取所有图像文件
    lr_path = f'./data/DIV2K_{mode}_LR'
    image_files = sorted([f for f in os.listdir(lr_path) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # 处理每张图像
    for img_name in tqdm(image_files, desc="处理进度"):
        # 读取并预处理图像
        img_path = os.path.join(lr_path, img_name)
        lr_img = Image.open(img_path).convert('RGB')
        lr_tensor = transform(lr_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 使用自集成策略
            sr_dat = SelfEnsembleStrategy(dat_model)(lr_tensor)
            sr_swinir = SelfEnsembleStrategy(swinir_model)(lr_tensor)
            sr_rfdn = SelfEnsembleStrategy(rfdn_model)(lr_tensor)
            
            # 融合处理
            combined = torch.cat([sr_dat, sr_swinir, sr_rfdn], dim=1)
            weights = weight_net(combined)
            
            # 获取权重并融合
            w1, w2, w3 = weights.chunk(3, dim=1)
            fused = w1 * sr_dat + w2 * sr_swinir + w3 * sr_rfdn
            fused = torch.clamp(fused, 0, 1)
        
        # 保存结果
        save_path = os.path.join(save_dir, f'{img_name.split(".")[0]}x4.png')
        save_image(fused, save_path)
        
        # 打印当前图像的权重信息
        print(f"\n处理图片: {img_name}")
        print(f"权重分布 - DAT: {w1.mean().item():.3f}, "
              f"SwinIR: {w2.mean().item():.3f}, "
              f"RFDN: {w3.mean().item():.3f}")
        print("-" * 50)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # 设置要处理的数据集
    mode = 'test'  # 或 'test'
    inference(mode)
