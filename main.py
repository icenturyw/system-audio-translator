import os
import sys
import site
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# 尝试通过模块导入来定位路径，这是最准确的方法
try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
    
    nvidia_paths = [
        os.path.dirname(nvidia.cublas.lib.__file__),
        os.path.dirname(nvidia.cudnn.lib.__file__)
    ]
except ImportError:
    # Fallback: manually construct path based on venv structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_site_packages = os.path.join(base_dir, 'venv', 'Lib', 'site-packages')
    nvidia_paths = [
        os.path.join(venv_site_packages, "nvidia", "cublas", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cublas", "lib"),
        os.path.join(venv_site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cudnn", "lib")
    ]

# 还有一种情况: site-packages 可能不在 venv 下 (取决于你是如何运行的)
# 让我们打印一下最终添加了什么，方便调试
print(f"{Fore.CYAN}正在配置 CUDA 环境...{Style.RESET_ALL}")
cuda_ok = False
for path in nvidia_paths:
    if os.path.exists(path):
        try:
            os.add_dll_directory(path)
            # print(f"  已添加 DLL 路径: {path}")
            cuda_ok = True
        except Exception as e:
            print(f"  添加路径失败: {path} ({e})")

if not cuda_ok:
    print(f"{Fore.YELLOW}警告: 未找到 NVIDIA CUDA/cuDNN DLL 路径。GPU 加速可能无法使用。{Style.RESET_ALL}")
else:
    print(f"{Fore.GREEN}CUDA 环境配置完成。{Style.RESET_ALL}")

import pyaudiowpatch as pyaudio
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import queue
import time
import argparse
import threading
from scipy import signal

class RealTimeTranslator:
    def __init__(self, model_size="tiny", device="auto", compute_type="int8", source_type="mic", target_lang="zh-CN"):
        print(f"{Fore.CYAN}正在初始化模型 ({model_size})... 这可能需要几分钟...{Style.RESET_ALL}")
        
        # 1. 初始化模型
        try:
            if device == "auto":
                # 强制尝试 CUDA 以获取详细报错
                print(f"{Fore.CYAN}正在尝试加载 CUDA 后端...{Style.RESET_ALL}")
                try:
                    self.model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
                    device = "cuda"
                except Exception as cuda_error:
                    print(f"{Fore.RED}CUDA 初始化失败: {cuda_error}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}正在回退到 CPU 模式...{Style.RESET_ALL}")
                    device = "cpu"
                    self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
            else:
                 self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            
            print(f"{Fore.GREEN}使用计算设备: {device}{Style.RESET_ALL}")
            self.translator = GoogleTranslator(source='auto', target=target_lang)
            print(f"{Fore.GREEN}模型加载完成! (目标语言: {target_lang}){Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}模型加载失败: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}提示: 如果是 CUDA 相关错误，请检查是否安装了 cuDNN 和 zlib。{Style.RESET_ALL}")
            sys.exit(1)

        self.q = queue.Queue()
        self.target_samplerate = 16000 # Whisper 期望
        self.chunk_size = 4096 
        
        self.source_type = source_type
        self.p = pyaudio.PyAudio()
        self.input_device_index = None
        self.input_channels = 1 
        self.device_samplerate = 16000 
        self.is_loopback = False

        # 2. 配置音频输入设备
        if source_type == "system":
            print(f"{Fore.CYAN}正在搜索 Windows WASAPI Loopback 设备...{Style.RESET_ALL}")
            try:
                # 获取默认的 WASAPI Loopback 设备
                wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                
                if not default_speakers["isLoopbackDevice"]:
                    for loopback in self.p.get_loopback_device_info_generator():
                        if default_speakers["name"] in loopback["name"]:
                            default_speakers = loopback
                            break
                    else:
                         print(f"{Fore.YELLOW}未找到完全匹配的 Loopback，使用第一个可用设备...{Style.RESET_ALL}")
                         for loopback in self.p.get_loopback_device_info_generator():
                             default_speakers = loopback
                             break

                print(f"{Fore.GREEN}已选择系统内录设备: {default_speakers['name']}{Style.RESET_ALL}")
                print(f"原生采样率: {int(default_speakers['defaultSampleRate'])} Hz")
                
                self.input_device_index = default_speakers["index"]
                self.input_channels = default_speakers["maxInputChannels"]
                self.device_samplerate = int(default_speakers["defaultSampleRate"])
                self.is_loopback = True
                
            except Exception as e:
                print(f"{Fore.RED}无法配置系统内录: {e}{Style.RESET_ALL}")
                sys.exit(1)
        else:
            print(f"{Fore.CYAN}使用默认麦克风输入...{Style.RESET_ALL}")
            info = self.p.get_default_input_device_info()
            self.input_device_index = info["index"]
            self.input_channels = 1
            self.device_samplerate = int(info["defaultSampleRate"])

        # VAD / Recording state
        self.is_recording = False
        self.audio_buffer = []
        self.silence_threshold = 0.015
        self.silence_duration = 0 
        self.silence_limit = 20    
        self.running = True

    def _check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback."""
        if status:
            pass
        self.q.put(in_data)
        return (None, pyaudio.paContinue)

    def process_audio(self):
        """Main loop."""
        msg = "系统音频" if self.source_type == "system" else "麦克风"
        print(f"{Fore.YELLOW}>>> 开始监听 [{msg}]... (按 Ctrl+C 停止){Style.RESET_ALL}")
        print("-" * 50)

        try:
            stream = self.p.open(format=pyaudio.paInt16,
                                 channels=self.input_channels,
                                 rate=self.device_samplerate, 
                                 input=True,
                                 input_device_index=self.input_device_index,
                                 frames_per_buffer=self.chunk_size,
                                 stream_callback=self.callback)
            
            stream.start_stream()

            while self.running:
                try:
                    in_data = self.q.get()
                    audio_chunk = np.frombuffer(in_data, dtype=np.int16)
                    
                    audio_float = audio_chunk.astype(np.float32) / 32768.0

                    if self.input_channels > 1:
                        audio_float = audio_float.reshape(-1, self.input_channels)
                        audio_float = np.mean(audio_float, axis=1)

                    if self.device_samplerate != self.target_samplerate:
                        num_samples = int(len(audio_float) * self.target_samplerate / self.device_samplerate)
                        audio_float = signal.resample(audio_float, num_samples)

                    rms = np.sqrt(np.mean(audio_float**2))
                    
                    if rms > 0.01: 
                        if not self.is_recording:
                            print(f"{Fore.BLUE}[检测到声音...]{Style.RESET_ALL}", end="\r")
                            self.is_recording = True
                        self.silence_duration = 0
                        self.audio_buffer.append(audio_float)
                    else:
                        if self.is_recording:
                            self.silence_duration += 1
                            self.audio_buffer.append(audio_float)
                            
                            if self.silence_duration >= 25: 
                                self.transcribe_and_translate()
                                self.reset_recording()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Error in loop: {e}")
                    break
                    
            stream.stop_stream()
            stream.close()
            self.p.terminate()

        except Exception as e:
            print(f"\n{Fore.RED}音频设备初始化失败!{Style.RESET_ALL}")
            print(f"错误信息: {e}")

    def reset_recording(self):
        self.is_recording = False
        self.audio_buffer = []
        self.silence_duration = 0

    def transcribe_and_translate(self):
        if not self.audio_buffer:
            return

        print(f"{Fore.MAGENTA}正在翻译...          {Style.RESET_ALL}", end="\r")
        
        audio_data = np.concatenate(self.audio_buffer, axis=0)
        
        if len(audio_data) < self.target_samplerate * 0.5:
            return

        try:
            segments, info = self.model.transcribe(audio_data, beam_size=5)
            
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
            
            full_text = full_text.strip()
            
            if full_text:
                translated = self.translator.translate(full_text)
                print(f"{Fore.CYAN}原文 ({info.language}): {full_text}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}译文: {translated}{Style.RESET_ALL}")
                print("-" * 30)
        except Exception as e:
            print(f"{Fore.RED}处理出错: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python 实时同声传译")
    parser.add_argument("--system", action="store_true", help="监听系统音频 (Loopback)")
    parser.add_argument("--model", default="tiny", help="Whisper 模型大小")
    parser.add_argument("--target", default="zh-CN", help="目标语言代码 (例如: zh-CN, hi, en, ja)")
    
    args = parser.parse_args()
    
    import shutil
    if not shutil.which("ffmpeg"):
        print(f"{Fore.RED}警告: 未找到 FFmpeg!{Style.RESET_ALL}")

    source = "system" if args.system else "mic"
    
    try:
        app = RealTimeTranslator(model_size=args.model, source_type=source, target_lang=args.target)
        app.process_audio()
    except KeyboardInterrupt:
        print("\n程序已退出。")