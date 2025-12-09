# 🎙️ AI Real-time Translator (AI 实时同声传译)

![App Screenshot](example.png)

一个基于 Python 的高性能实时同声传译工具。支持**监听电脑系统声音**（如会议、电影、YouTube）或**麦克风输入**，利用 OpenAI 的 Whisper 模型进行语音识别，并实时翻译成目标语言。

本项目拥有现代化的 **Dark Mode GUI**，支持 **NVIDIA GPU 加速**，并针对连续语音进行了断句优化。

## ✨ 核心功能

*   **🎧 系统音频内录 (Windows Loopback)**: 直接捕获电脑发出的声音，无需麦克风中转，非常适合观看生肉视频、Zoom/Teams 会议。
*   **🎙️ 麦克风监听**: 支持常规的麦克风实时翻译。
*   **🚀 GPU 加速 (CUDA)**: 集成 `faster-whisper` 和自动 CUDA 环境配置，在 NVIDIA 显卡上实现毫秒级响应。
*   **⚡ 智能 VAD 断句**: 针对连续说话场景（如新闻、演讲）设计了动态静音检测和强制切分逻辑，告别无限等待。
*   **🌏 多语言支持**: 自动检测源语言（支持印地语、英语、日语等几乎所有语言），默认翻译为中文（可配置）。
*   **🎨 现代化 GUI**: 基于 `CustomTkinter` 构建，支持高分屏，美观的深色模式。

## 🛠️ 技术栈

*   **GUI**: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
*   **ASR (语音识别)**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)
*   **Translation**: Google Translate API
*   **Audio**: `pyaudiowpatch` (支持 WASAPI Loopback) + `scipy` (重采样)

## 📦 安装指南

### 1. 环境要求
*   Windows 10/11 (系统内录功能依赖 WASAPI)
*   Python 3.8+
*   **FFmpeg** (必须安装)
*   NVIDIA 显卡 (推荐，用于 GPU 加速)

### 2. 安装 FFmpeg
Whisper 依赖 FFmpeg 处理音频。在 PowerShell 中运行：
```powershell
winget install Gyan.FFmpeg
# 安装后请重启终端或电脑以生效
```

### 3. 克隆与安装依赖
```powershell
git clone https://github.com/your-username/translator.git
cd translator

# 创建虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1

# 安装依赖 (会自动安装 CUDA 相关的 Python 库)
pip install -r requirements.txt
pip install customtkinter pyaudiowpatch faster-whisper deep-translator scipy colorama soundfile numpy
# 或者直接安装 NVIDIA 库以确保 GPU 可用
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## 🚀 使用方法

### 启动 GUI
```powershell
python gui.py
```

### 操作指南
1.  **模型大小**: 
    *   `tiny/base`: 速度极快，精度一般。
    *   `small/medium`: 推荐，速度与精度平衡（截图中使用的是 small）。
    *   `large-v3`: 精度最高，但需要较好的显卡 (8GB+ VRAM)。
2.  **输入源**:
    *   `System Audio (系统)`: 录制电脑正在播放的声音。
    *   `Microphone (麦克风)`: 录制你的声音。
3.  点击 **"启动监听"** 即可开始。

> **首次运行提示**: 第一次选择某个模型时，程序会自动从 HuggingFace 下载模型权重，界面可能会短暂显示“加载中”，请耐心等待。

## ⚙️ 常见问题 (FAQ)

**Q: 为什么显示 "使用计算设备: cpu"？**
A: 请确保你安装了 NVIDIA 显卡驱动。本项目内置了自动查找 `nvidia-cudnn` 库的逻辑，通常无需手动配置 CUDA 环境变量。如果依然失败，请检查 `pip list` 是否包含 `nvidia-cudnn-cu12`。

**Q: 翻译有延迟？**
A: 
1. 确保使用了 GPU 加速。
2. 尝试使用更小的模型（如 `small` 代替 `medium`）。
3. 程序内置了 8秒 强制断句逻辑，连续说话时会有轻微缓冲是正常的。

**Q: 报错 `[Errno -9997] Invalid sample rate`?**
A: 本项目已内置自动重采样算法 (`scipy.signal.resample`)，会自动将系统音频（通常 48kHz）转换为 Whisper 需要的 16kHz。如果报错，请确保没有其他程序独占音频设备。

## 📜 许可证

MIT License
