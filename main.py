import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, ttk
import torch.nn.functional as F
import os
import urllib.request
from urllib.error import URLError
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import psutil
import tempfile
from PIL import Image, ImageGrab

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("digit_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 检测可用硬件资源
def get_hardware_info():
    # CPU信息
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    
    # 内存信息
    mem = psutil.virtual_memory()
    total_mem = mem.total / (1024**3)  # GB
    
    # GPU信息
    gpu_info = "未检测到GPU"
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            gpu_memory = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count)]
            gpu_info = f"检测到 {gpu_count} 个GPU: {', '.join([f'{name} ({mem:.1f}GB)' for name, mem in zip(gpu_names, gpu_memory)])}"
    except:
        pass
    
    return {
        "cpu_count": cpu_count,
        "cpu_count_logical": cpu_count_logical,
        "total_mem": total_mem,
        "gpu_info": gpu_info
    }

# 根据硬件配置自动调整训练参数
def auto_tune_parameters(hardware_info, epochs=None):
    params = {
        'batch_size': 256,
        'num_workers': min(8, hardware_info['cpu_count']),
        'mixed_precision': True,
        'gradient_checkpointing': False,
        'use_amp': True,
        'epochs': epochs if epochs is not None else 3,  # 默认3轮，可由用户指定
        'lr': 0.001
    }
    
    # 根据GPU内存调整batch_size
    if '4090' in hardware_info['gpu_info']:
        params['batch_size'] = 512  # RTX 4090可以处理更大的batch_size
    
    # 如果内存充足，增加数据加载器的工作进程
    if hardware_info['total_mem'] > 32:
        params['num_workers'] = min(12, hardware_info['cpu_count'])
    
    return params

# 定义改进的ResNet类
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

# 自定义进度条回调函数
class DownloadProgressBar(tk.Toplevel):
    def __init__(self, parent, title, max_val):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x100")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.label = tk.Label(self, text="准备下载...", font=("SimHei", 10))
        self.label.pack(pady=10)
        
        self.progress = ttk.Progressbar(self, orient="horizontal", length=350, mode="determinate")
        self.progress['maximum'] = max_val
        self.progress.pack(pady=10)
        
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (parent.winfo_width() // 2) - (width // 2) + parent.winfo_x()
        y = (parent.winfo_height() // 2) - (height // 2) + parent.winfo_y()
        self.geometry(f"+{x}+{y}")
        
        self.current_val = 0
        
    def update(self, count, block_size, total_size):
        if total_size > 0:
            downloaded = count * block_size
            if downloaded < total_size:
                self.progress['value'] = downloaded
                self.label.config(text=f"下载中: {downloaded/1024/1024:.2f} MB / {total_size/1024/1024:.2f} MB")
            else:
                self.progress['value'] = total_size
                self.label.config(text=f"下载完成: {total_size/1024/1024:.2f} MB")
            self.update_idletasks()

# 训练进度窗口
class TrainingProgressWindow(tk.Toplevel):
    def __init__(self, parent, epochs, steps_per_epoch):
        super().__init__(parent)
        self.title("模型训练进度")
        self.geometry("400x150")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.epoch_label = tk.Label(self, text="准备训练...", font=("SimHei", 10))
        self.epoch_label.pack(pady=5)
        
        self.epoch_frame = tk.Frame(self)
        self.epoch_frame.pack(fill="x", padx=20, pady=5)
        
        self.epoch_text = tk.Label(self.epoch_frame, text="Epoch:", font=("SimHei", 10))
        self.epoch_text.pack(side=tk.LEFT)
        
        self.epoch_progress = ttk.Progressbar(self.epoch_frame, orient="horizontal", length=280, mode="determinate")
        self.epoch_progress['maximum'] = epochs
        self.epoch_progress.pack(side=tk.LEFT, padx=10)
        
        self.epoch_count = tk.Label(self.epoch_frame, text="0/0", font=("SimHei", 10))
        self.epoch_count.pack(side=tk.LEFT)
        
        self.step_frame = tk.Frame(self)
        self.step_frame.pack(fill="x", padx=20, pady=5)
        
        self.step_text = tk.Label(self.step_frame, text="步骤:", font=("SimHei", 10))
        self.step_text.pack(side=tk.LEFT)
        
        self.step_progress = ttk.Progressbar(self.step_frame, orient="horizontal", length=280, mode="determinate")
        self.step_progress['maximum'] = steps_per_epoch
        self.step_progress.pack(side=tk.LEFT, padx=10)
        
        self.step_count = tk.Label(self.step_frame, text="0/0", font=("SimHei", 10))
        self.step_count.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(self, text="", font=("SimHei", 10))
        self.status_label.pack(pady=5)
        
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (parent.winfo_width() // 2) - (width // 2) + parent.winfo_x()
        y = (parent.winfo_height() // 2) - (height // 2) + parent.winfo_y()
        self.geometry(f"+{x}+{y}")
        
        self.current_epoch = 0
        self.current_step = 0
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        self.epoch_progress['value'] = epoch
        self.epoch_count.config(text=f"{epoch}/{self.epochs}")
        self.epoch_label.config(text=f"Epoch {epoch}/{self.epochs}")
        self.step_progress['value'] = 0
        self.step_count.config(text="0/0")
        self.update_idletasks()
        
    def update_step(self, step, loss, acc):
        self.current_step = step
        self.step_progress['value'] = step
        self.step_count.config(text=f"{step}/{self.steps_per_epoch}")
        self.status_label.config(text=f"损失: {loss:.4f} | 准确率: {acc:.2f}%")
        self.update_idletasks()
        
    def update_validation(self, loss, acc):
        self.status_label.config(text=f"验证损失: {loss:.4f} | 验证准确率: {acc:.2f}%")
        self.update_idletasks()

# 轮次选择对话框
class EpochSelectionDialog(tk.Toplevel):
    def __init__(self, parent, default_epochs=3):
        super().__init__(parent)
        self.title("选择训练轮次")
        self.geometry("300x150")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.epochs = default_epochs
        self.result = None
        
        # 窗口居中
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (parent.winfo_width() // 2) - (width // 2) + parent.winfo_x()
        y = (parent.winfo_height() // 2) - (height // 2) + parent.winfo_y()
        self.geometry(f"+{x}+{y}")
        
        # 创建界面
        ttk.Label(self, text="请选择训练轮次:", font=("SimHei", 12)).pack(pady=15)
        
        # 轮次选择框
        self.epoch_var = tk.StringVar(value=str(default_epochs))
        ttk.Combobox(self, textvariable=self.epoch_var, values=[str(i) for i in range(1, 21)], width=10).pack(pady=5)
        
        # 按钮框
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15, fill=tk.X, padx=20)
        
        ttk.Button(btn_frame, text="确定", command=self.on_ok).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="取消", command=self.on_cancel).pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
    def on_ok(self):
        try:
            self.epochs = int(self.epoch_var.get())
            self.result = self.epochs
            self.destroy()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的轮次数")
    
    def on_cancel(self):
        self.destroy()

# 加载预训练模型
def load_model(root=None, epochs=None):
    try:
        model = ResNet()
        
        if not os.path.exists('mnist_model.pth'):
            raise FileNotFoundError("未找到预训练模型文件")
            
        logger.info("加载预训练模型...")
        state_dict = torch.load('mnist_model.pth')
        
        try:
            model.load_state_dict(state_dict)
            logger.info('模型加载成功')
            model.eval()
            return model
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e) or "Unexpected key(s) in state_dict" in str(e):
                logger.warning("检测到模型架构不匹配")
                logger.warning("请删除'mnist_model.pth'文件，程序将重新训练一个新模型")
                raise FileNotFoundError("模型架构不匹配")
            else:
                raise e
                
    except FileNotFoundError:
        logger.info('未找到预训练模型或模型架构不匹配，开始训练新模型...')
        try:
            return train_model(root, epochs)
        except RuntimeError as e:
            logger.error(f"训练模型失败: {e}")
            print("\n=====================================")
            print("MNIST数据集下载失败或已损坏。")
            print("请按以下步骤手动下载数据集:")
            print("1. 访问: http://yann.lecun.com/exdb/mnist/")
            print("2. 下载以下4个文件:")
            print("   - train-images-idx3-ubyte.gz")
            print("   - train-labels-idx1-ubyte.gz")
            print("   - t10k-images-idx3-ubyte.gz")
            print("   - t10k-labels-idx1-ubyte.gz")
            print("3. 创建目录: ./data/MNIST/raw/")
            print("4. 将下载的文件放入该目录中")
            print("5. 重新运行程序")
            print("=====================================\n")
            exit(1)

# 训练模型 - 针对RTX 4090优化
def train_model(root=None, epochs=None):
    # 获取硬件信息并自动调整参数
    hardware_info = get_hardware_info()
    params = auto_tune_parameters(hardware_info, epochs)
    
    logger.info(f"硬件配置: {hardware_info['gpu_info']}, CPU核心: {hardware_info['cpu_count']}, 内存: {hardware_info['total_mem']:.1f}GB")
    logger.info(f"自动调整的训练参数: {params}")
    
    # 创建数据目录
    os.makedirs('./data/MNIST/raw', exist_ok=True)
    
    # 增强的数据预处理
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 尝试下载MNIST数据集
    logger.info("准备下载MNIST数据集...")
    
    # 创建下载进度窗口
    if root:
        download_window = DownloadProgressBar(root, "下载MNIST数据集", 17970600)  # MNIST数据集大小约17MB
    else:
        download_window = None
    
    try:
        # 尝试使用不同的镜像源
        mirror_urls = [
            'https://ossci-datasets.s3.amazonaws.com/mnist/',
            'http://yann.lecun.com/exdb/mnist/'
        ]
        
        success = False
        for mirror in mirror_urls:
            try:
                logger.info(f"尝试从镜像 {mirror} 下载数据集...")
                
                # 下载训练图像
                train_images_url = f"{mirror}train-images-idx3-ubyte.gz"
                train_images_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
                
                if not os.path.exists(train_images_path):
                    if download_window:
                        urllib.request.urlretrieve(train_images_url, train_images_path, reporthook=download_window.update)
                    else:
                        urllib.request.urlretrieve(train_images_url, train_images_path)
                
                # 下载训练标签
                train_labels_url = f"{mirror}train-labels-idx1-ubyte.gz"
                train_labels_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
                
                if not os.path.exists(train_labels_path):
                    if download_window:
                        urllib.request.urlretrieve(train_labels_url, train_labels_path, reporthook=download_window.update)
                    else:
                        urllib.request.urlretrieve(train_labels_url, train_labels_path)
                
                # 下载测试图像
                test_images_url = f"{mirror}t10k-images-idx3-ubyte.gz"
                test_images_path = './data/MNIST/raw/t10k-images-idx3-ubyte.gz'
                
                if not os.path.exists(test_images_path):
                    if download_window:
                        urllib.request.urlretrieve(test_images_url, test_images_path, reporthook=download_window.update)
                    else:
                        urllib.request.urlretrieve(test_images_url, test_images_path)
                
                # 下载测试标签
                test_labels_url = f"{mirror}t10k-labels-idx1-ubyte.gz"
                test_labels_path = './data/MNIST/raw/t10k-labels-idx1-ubyte.gz'
                
                if not os.path.exists(test_labels_path):
                    if download_window:
                        urllib.request.urlretrieve(test_labels_url, test_labels_path, reporthook=download_window.update)
                    else:
                        urllib.request.urlretrieve(test_labels_url, test_labels_path)
                
                # 创建数据集
                train_dataset = datasets.MNIST('./data', train=True, download=False, transform=train_transform)
                test_dataset = datasets.MNIST('./data', train=False, download=False, transform=test_transform)
                
                success = True
                break
            except URLError as e:
                logger.warning(f"从 {mirror} 下载失败: {e}")
                continue
        
        if not success:
            raise RuntimeError("所有镜像源下载失败")
    
    except Exception as e:
        # 清理可能不完整的下载
        for f in os.listdir('./data/MNIST/raw'):
            if f.endswith('.gz') or f.endswith('.ubyte'):
                os.remove(os.path.join('./data/MNIST/raw', f))
        
        if download_window:
            download_window.destroy()
            
        raise RuntimeError(f"下载MNIST数据集失败: {e}")
    
    if download_window:
        download_window.destroy()
    
    # 创建数据加载器 - 使用自动调整的batch_size和num_workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True, 
        num_workers=params['num_workers'], 
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=params['batch_size'], 
        shuffle=False, 
        num_workers=params['num_workers'], 
        pin_memory=True,
        persistent_workers=True
    )
    
    # 初始化模型
    model = ResNet()
    
    # 使用梯度检查点减少内存占用
    if params['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
    
    # 设置训练设备 - 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 优化模型在GPU上的内存使用和计算效率
    if device.type == "cuda":
        # 使用半精度训练以提高速度
        if params['mixed_precision']:
            model = model.to(device, memory_format=torch.channels_last).half()
        else:
            model = model.to(device, memory_format=torch.channels_last)
            
        # 启用cuDNN benchmark以优化卷积算法
        torch.backends.cudnn.benchmark = True
        
        # 打印GPU内存信息
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU总内存: {total_gpu_mem:.2f} GB")
    else:
        model = model.to(device)
    
    # 初始化优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.3)
    criterion = nn.NLLLoss()
    
    # 混合精度训练设置
    scaler = torch.cuda.amp.GradScaler(enabled=params['use_amp'])
    
    # 创建训练进度窗口
    if root:
        training_window = TrainingProgressWindow(root, params['epochs'], len(train_loader))
    else:
        training_window = None
    
    # 训练模型
    logger.info(f"开始训练模型，总轮次: {params['epochs']}，设备: {device}")
    best_acc = 0.0
    
    with logging_redirect_tqdm():
        for epoch in range(1, params['epochs'] + 1):
            if training_window:
                training_window.update_epoch(epoch)
            
            # 训练阶段
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            # 使用tqdm显示训练进度
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}/{params["epochs"]}') as pbar:
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    # 将数据移至设备
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    # 混合精度训练
                    with torch.cuda.amp.autocast(enabled=params['use_amp']):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True节省内存
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # 计算当前批次的损失和准确率
                    batch_loss = loss.item()
                    batch_acc = 100. * correct / total
                    
                    # 更新进度条和日志
                    pbar.set_postfix({'loss': batch_loss, 'acc': batch_acc})
                    pbar.update(1)
                    
                    # 更新GUI进度窗口
                    if training_window:
                        training_window.update_step(batch_idx + 1, batch_loss, batch_acc)
            
            # 计算平均训练损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = 100. * correct / total
            logger.info(f'Epoch {epoch}/{params["epochs"]} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%')
            
            # 验证阶段
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast(enabled=params['use_amp']):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            # 计算平均验证损失和准确率
            avg_test_loss = test_loss / len(test_loader)
            avg_test_acc = 100. * correct / total
            logger.info(f'Epoch {epoch}/{params["epochs"]} | Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.2f}%')
            
            # 更新学习率
            scheduler.step(avg_test_loss)
            
            # 更新进度窗口
            if training_window:
                training_window.update_validation(avg_test_loss, avg_test_acc)
            
            # 保存最佳模型
            if avg_test_acc > best_acc:
                best_acc = avg_test_acc
                torch.save(model.state_dict(), 'mnist_model.pth')
                logger.info(f'模型已保存，准确率: {best_acc:.2f}%')
    
    if training_window:
        training_window.destroy()
    
    # 加载最佳模型
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    return model

# 数字识别应用类
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别系统")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置中文字体
        self.style = ttk.Style()
        self.style.configure('.', font=('SimHei', 10))
        
        # 加载模型
        self.model = load_model(root)
        
        # 创建界面
        self.create_widgets()
        
        # 初始化画布
        self.init_canvas()
        
        # 显示欢迎信息
        self.display_welcome_message()
    
    def create_widgets(self):
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧框架 - 画布和按钮
        self.left_frame = ttk.Frame(self.main_frame, width=400, height=500)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 右侧框架 - 结果和信息
        self.right_frame = ttk.Frame(self.main_frame, width=350, height=500)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 画布区域
        self.canvas_frame = ttk.LabelFrame(self.left_frame, text="手写数字", padding="10")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建画布
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=300, height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 按钮区域
        self.button_frame = ttk.Frame(self.left_frame, padding="10")
        self.button_frame.pack(fill=tk.X, pady=10)
        
        # 识别按钮
        self.recognize_btn = ttk.Button(self.button_frame, text="识别数字", command=self.recognize_digit)
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        
        # 清除按钮
        self.clear_btn = ttk.Button(self.button_frame, text="清除画布", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 重新训练按钮
        self.retrain_btn = ttk.Button(self.button_frame, text="重新训练模型", command=self.retrain_model)
        self.retrain_btn.pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        self.result_frame = ttk.LabelFrame(self.right_frame, text="识别结果", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 数字显示标签
        self.digit_label = ttk.Label(self.result_frame, text="?", font=('SimHei', 72, 'bold'))
        self.digit_label.pack(pady=20)
        
        # 置信度显示
        self.confidence_label = ttk.Label(self.result_frame, text="置信度: --%", font=('SimHei', 12))
        self.confidence_label.pack(pady=10)
        
        # 预测概率分布图
        self.prob_frame = ttk.LabelFrame(self.right_frame, text="预测概率分布", padding="10")
        self.prob_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建图表
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=self.prob_frame)
        self.canvas_chart.draw()
        self.canvas_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def init_canvas(self):
        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_paint)
    
    def start_paint(self, event):
        self.last_x, self.last_y = event.x, event.y
    
    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line((self.last_x, self.last_y, event.x, event.y), 
                                   width=15, fill="black", capstyle=tk.ROUND, smooth=True)
        self.last_x, self.last_y = event.x, event.y
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.digit_label.config(text="?")
        self.confidence_label.config(text="置信度: --%")
        self.clear_chart()
    
    def clear_chart(self):
        self.ax.clear()
        self.ax.set_xlabel('数字')
        self.ax.set_ylabel('概率 (%)')
        self.ax.set_title('预测概率分布')
        self.ax.set_xticks(range(10))
        self.ax.set_ylim(0, 100)
        self.canvas_chart.draw()
    
    def display_welcome_message(self):
        self.clear_chart()
        self.digit_label.config(text="?")
        self.confidence_label.config(text="请在左侧画布上书写数字")
    
    def preprocess_image(self):
        """使用PNG方案替代EPS，避免依赖Ghostscript"""
        # 创建临时PNG文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # 将画布内容保存为PNG
        try:
            # 获取画布在屏幕上的位置和大小
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            
            # 截取画布区域
            img = ImageGrab.grab((x, y, x + width, y + height))
            img.save(temp_path)
            
            # 打开并处理图像
            img = plt.imread(temp_path)
            
        finally:
            # 确保删除临时文件
            os.unlink(temp_path)
        
        # 转换为灰度图并调整大小
        if len(img.shape) > 2:
            img = np.mean(img[:, :, :3], axis=2)  # 转换为灰度
            
        # 反转颜色（MNIST中数字为白色，背景为黑色）
        img = 1.0 - img
        
        # 调整大小为28x28
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize((28, 28), Image.LANCZOS)
        img = np.array(img) / 255.0
        
        # 标准化
        img = (img - 0.1307) / 0.3081
        
        # 转换为PyTorch张量
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
    
    def recognize_digit(self):
        # 预处理图像
        img_tensor = self.preprocess_image()
        
        # 将图像移至设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)
        
        # 如果模型是半精度的，也将输入转为半精度
        if next(self.model.parameters()).dtype == torch.float16:
            img_tensor = img_tensor.half()
        
        # 进行预测
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.exp(output)
            conf, pred = probs.max(1)
        
        # 显示结果
        self.digit_label.config(text=str(pred.item()))
        self.confidence_label.config(text=f"置信度: {conf.item()*100:.2f}%")
        
        # 更新概率分布图
        self.update_prob_chart(probs.cpu().numpy()[0] * 100)
    
    def update_prob_chart(self, probs):
        self.ax.clear()
        digits = range(10)
        bars = self.ax.bar(digits, probs, color='skyblue')
        
        # 高亮预测的数字
        predicted_digit = np.argmax(probs)
        bars[predicted_digit].set_color('red')
        
        # 添加数据标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{probs[i]:.1f}%', ha='center', va='bottom')
        
        self.ax.set_xlabel('数字')
        self.ax.set_ylabel('概率 (%)')
        self.ax.set_title('预测概率分布')
        self.ax.set_xticks(digits)
        self.ax.set_ylim(0, 110)
        self.canvas_chart.draw()
    
    def retrain_model(self):
        # 打开轮次选择对话框
        dialog = EpochSelectionDialog(self.root)
        self.root.wait_window(dialog)
        
        # 检查用户是否选择了轮次
        if dialog.result is not None:
            epochs = dialog.result
            
            # 确认对话框
            answer = messagebox.askyesno("确认", f"确定要重新训练模型吗？将训练 {epochs} 轮，这可能需要一些时间。")
            if answer:
                # 禁用按钮
                self.recognize_btn.config(state=tk.DISABLED)
                self.clear_btn.config(state=tk.DISABLED)
                self.retrain_btn.config(state=tk.DISABLED)
                
                # 显示训练中消息
                self.digit_label.config(text="训练中...")
                self.confidence_label.config(text="请等待...")
                self.clear_chart()
                
                # 刷新界面
                self.root.update()
                
                # 重新训练模型
                self.model = train_model(self.root, epochs)
                
                # 恢复按钮状态
                self.recognize_btn.config(state=tk.NORMAL)
                self.clear_btn.config(state=tk.NORMAL)
                self.retrain_btn.config(state=tk.NORMAL)
                
                # 显示训练完成消息
                messagebox.showinfo("训练完成", f"模型已完成 {epochs} 轮训练！")
                self.display_welcome_message()

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()