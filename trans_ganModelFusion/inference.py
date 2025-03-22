import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from train_fusion import FusionModel

def inference():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = FusionModel(device).to(device)
    model.load_state_dict(torch.load('./checkpoints/realESRGAN/best_pth/best_fusion.pth'))
    model.eval()
    
    # 创建结果保存目录
    os.makedirs('./trans_ganModelFusion/results_test', exist_ok=True)
    
    # 图像预处理
    to_tensor = transforms.ToTensor()
    
    # 获取测试图像路径
    test_dir = './data/DIV2K_test_LR'
    test_images = sorted(os.listdir(test_dir))
    
    with torch.no_grad():
        for img_name in test_images:
            # 读取LR图像
            lr_path = os.path.join(test_dir, img_name)
            lr_img = Image.open(lr_path).convert('RGB')
            lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
            
            # 获取权重预测
            weights = model.predictor(lr_tensor)  # [1, 3]
            weights = weights.squeeze().cpu().numpy()

            # 生成SR图像
            sr = model(lr_tensor)

            # 确保输出图像值在0到1之间
            sr = torch.clamp(sr, 0, 1)
            
            # 转换为PIL图像并保存
            sr_img = transforms.ToPILImage()(sr.squeeze().cpu())
            save_path = os.path.join('./trans_ganModelFusion/results_test', '{:4}x4.png'.format(img_name.split('.')[0]))
            sr_img.save(save_path)
            
            print(f'Processed: {img_name}')
            print(f'Model weights: RealESRGAN: {weights[0]:.3f}, DAT: {weights[1]:.3f}, RFDN: {weights[2]:.3f}')
            print('-' * 50)

if __name__ == '__main__':
    inference()
