# 手写数字识别系统

这是一个基于PyTorch和ResNet架构的手写数字识别应用，通过图形界面让用户能够直观地书写数字并获得实时识别结果。应用支持自定义训练轮次，自动优化训练参数以适应不同硬件配置，同时提供了友好的用户界面和详细的识别概率分布展示。

## 优点展示

- **直观的图形界面**：通过简单的画布界面，用户可以手写数字并立即获得识别结果
- **高精度识别**：采用优化的ResNet架构，在MNIST数据集上可达到99%以上的准确率
- **自定义训练**：支持用户自定义训练轮次，灵活调整模型性能
- **硬件自动优化**：根据系统硬件配置自动调整训练参数，充分利用GPU资源
- **详细的概率分布**：不仅显示识别结果，还提供对每个数字的预测概率分布
- **无需Ghostscript依赖**：采用PNG截图方案替代EPS，彻底解决依赖问题

## 前置依赖

该项目需要以下软件和库支持：

### 系统依赖
- Python 3.8 或更高版本
- 操作系统：Windows 10/11、Linux（Ubuntu 18.04+ 或类似）、macOS（10.15+）
- 推荐NVIDIA GPU（支持CUDA）以加速训练过程

### Python 库依赖
以下是主要依赖库及其版本要求：
```
torch >= 2.0.0
torchvision >= 0.15.0
matplotlib >= 3.7.1
numpy >= 1.24.3
Pillow >= 9.5.0
tkinter >= 8.6 (通常包含在Python标准库中)
psutil >= 5.9.5
tqdm >= 4.65.0
```

## 不同平台运行方法

### Windows 平台

1. **安装Python**
   - 从 [Python官网](https://www.python.org/downloads/windows/) 下载并安装Python 3.9或更高版本
   - 安装时确保勾选"Add Python to PATH"选项

2. **安装依赖库**
   - 打开命令提示符（CMD）或PowerShell
   - 执行以下命令安装所需库：
     ```
     pip install torch torchvision matplotlib numpy pillow psutil tqdm
     ```

3. **下载项目代码**
   - 从GitHub克隆项目仓库：
     ```
     git clone https://github.com/your-username/digit-recognizer.git
     cd digit-recognizer
     ```

4. **运行应用**
   - 在命令行中执行：
     ```
     python main.py
     ```

### Linux 平台

1. **安装Python和依赖库**
   - 对于Ubuntu/Debian系统：
     ```
     sudo apt-get update
     sudo apt-get install python3 python3-pip python3-tk
     pip3 install torch torchvision matplotlib numpy pillow psutil tqdm
     ```

2. **下载项目代码**
   - 克隆项目仓库：
     ```
     git clone https://github.com/your-username/digit-recognizer.git
     cd digit-recognizer
     ```

3. **运行应用**
   - 执行：
     ```
     python3 main.py
     ```

### macOS 平台

1. **安装Python**
   - 推荐使用Homebrew安装Python：
     ```
     brew install python3
     ```

2. **安装依赖库**
   - 执行：
     ```
     pip3 install torch torchvision matplotlib numpy pillow psutil tqdm
     ```

3. **下载项目代码**
   - 克隆项目仓库：
     ```
     git clone https://github.com/your-username/digit-recognizer.git
     cd digit-recognizer
     ```

4. **运行应用**
   - 执行：
     ```
     python3 main.py
     ```

### 模型指南

如果您的计算机性能较差，训练模型可能耗时较长，如配置不足以训练一个模型
请将项目文件夹下的 **/module/mnist_model.pth** 复制到 **/** 下 即可使用预训练模型
注意：预训练模型效果较差，请尽量选择自己训练模型，模型轮次建议设置为5以上（5轮次可达到95%精准度，13轮次可到达99%精准度）

### GPU加速支持（可选）

如果你的系统有NVIDIA GPU并希望加速训练过程，需要安装：
- CUDA Toolkit (对应你的GPU驱动版本)
- cuDNN库
- PyTorch的CUDA版本（安装时指定）

例如，安装支持CUDA 11.8的PyTorch：
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
