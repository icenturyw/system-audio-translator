import os
import sys
import site
from colorama import Fore, Style, init
import pyaudiowpatch as pyaudio
import numpy as np
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import queue
import threading
from scipy import signal
import time

# Initialize colorama
init(autoreset=True)

# Auto-configure CUDA paths
try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
    nvidia_paths = [
        os.path.dirname(nvidia.cublas.lib.__file__),
        os.path.dirname(nvidia.cudnn.lib.__file__)
    ]
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_site_packages = os.path.join(base_dir, 'venv', 'Lib', 'site-packages')
    nvidia_paths = [
        os.path.join(venv_site_packages, "nvidia", "cublas", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cublas", "lib"),
        os.path.join(venv_site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cudnn", "lib")
    ]

for path in nvidia_paths:
    if os.path.exists(path):
        try:
            os.add_dll_directory(path)
        except:
            pass

class TranslatorEngine:
    def __init__(self, model_size="tiny", device="auto", compute_type="int8", 
                 source_type="system", target_lang="zh-CN", 
                 on_subtitle=None, on_status=None):
        
        self.on_subtitle = on_subtitle # Callback(text, is_translation)
        self.on_status = on_status     # Callback(status_text)
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.source_type = source_type
        self.target_lang = target_lang
        
        self.running = False
        self.thread = None
        self.q = queue.Queue()
        self.model = None
        self.translator = None

    def initialize_model(self):
        if self.on_status: self.on_status(f"正在加载模型 ({self.model_size})...")
        
        try:
            if self.device == "auto":
                try:
                    self.model = WhisperModel(self.model_size, device="cuda", compute_type=self.compute_type)
                    self.device = "cuda"
                except Exception as e:
                    print(f"CUDA Error: {e}")
                    self.model = WhisperModel(self.model_size, device="cpu", compute_type=self.compute_type)
                    self.device = "cpu"
            else:
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

            self.translator = GoogleTranslator(source='auto', target=self.target_lang)
            if self.on_status: self.on_status(f"就绪 ({self.device})")
            return True
        except Exception as e:
            if self.on_status: self.on_status(f"模型加载失败: {e}")
            return False

    def start(self):
        if not self.model:
            if not self.initialize_model():
                return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
        if self.on_status: self.on_status("已停止")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        self.q.put(in_data)
        return (None, pyaudio.paContinue)

    def _process_audio_loop(self):
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            # Device Selection Logic
            device_index = None
            channels = 1
            rate = 16000
            
            if self.source_type == "system":
                if self.on_status: self.on_status("正在寻找系统音频设备...")
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                
                if not default_speakers["isLoopbackDevice"]:
                    for loopback in p.get_loopback_device_info_generator():
                        if default_speakers["name"] in loopback["name"]:
                            default_speakers = loopback
                            break
                    else:
                         for loopback in p.get_loopback_device_info_generator():
                             default_speakers = loopback
                             break
                
                device_index = default_speakers["index"]
                channels = default_speakers["maxInputChannels"]
                rate = int(default_speakers["defaultSampleRate"])
            else:
                info = p.get_default_input_device_info()
                device_index = info["index"]
                rate = int(info["defaultSampleRate"])

            if self.on_status: self.on_status(f"正在监听: {rate}Hz")

            stream = p.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=4096,
                            stream_callback=self._audio_callback)
            
            stream.start_stream()
            
            audio_buffer = []
            silence_frames = 0
            is_speaking = False
            total_frames = 0 # Track total frames in current buffer
            
            # Configuration
            MAX_DURATION_SECONDS = 8.0  # Force cut after 8 seconds of continuous speech
            MIN_SILENCE_FRAMES = 10     # ~0.5s default silence
            
            while self.running:
                try:
                    in_data = self.q.get(timeout=1)
                except queue.Empty:
                    continue

                audio_chunk = np.frombuffer(in_data, dtype=np.int16)
                audio_float = audio_chunk.astype(np.float32) / 32768.0

                if channels > 1:
                    audio_float = audio_float.reshape(-1, channels)
                    audio_float = np.mean(audio_float, axis=1)

                # Resample if needed
                if rate != 16000:
                    num_samples = int(len(audio_float) * 16000 / rate)
                    audio_float = signal.resample(audio_float, num_samples)

                rms = np.sqrt(np.mean(audio_float**2))
                
                # Approximate duration of one chunk (at 16k)
                # chunk_duration = len(audio_float) / 16000.0 
                # usually 4096 samples @ 48k -> resampled ~1365 samples @ 16k -> ~0.085s
                
                if rms > 0.01:
                    if not is_speaking:
                        is_speaking = True
                        if self.on_status: self.on_status("正在收听...")
                    silence_frames = 0
                    audio_buffer.append(audio_float)
                    total_frames += 1
                else:
                    if is_speaking:
                        silence_frames += 1
                        audio_buffer.append(audio_float)
                        total_frames += 1
                        
                        # Dynamic Silence Threshold:
                        # If buffer is long (>5s), be aggressive and cut on short silence (0.5s)
                        # If buffer is short, wait for longer silence (1.0s) to avoid cutting words
                        current_duration = total_frames * (len(audio_float) / 16000.0)
                        
                        dynamic_limit = 10 if current_duration > 4.0 else 20
                        
                        if silence_frames > dynamic_limit: 
                            self._transcribe(audio_buffer)
                            audio_buffer = []
                            is_speaking = False
                            silence_frames = 0
                            total_frames = 0
                            if self.on_status: self.on_status("等待语音...")
                
                # FORCE CUT logic
                # If speaker never stops, cut anyway to prevent lag
                current_duration = total_frames * (len(audio_float) / 16000.0)
                if is_speaking and current_duration > MAX_DURATION_SECONDS:
                     if self.on_status: self.on_status("强制断句翻译...")
                     self._transcribe(audio_buffer)
                     audio_buffer = []
                     # Don't reset is_speaking, effectively "splitting" the stream
                     # But we reset counters for the new segment
                     silence_frames = 0
                     total_frames = 0
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            if self.on_status: self.on_status(f"音频错误: {e}")
            print(e)

    def _transcribe(self, buffer):
        if not buffer: return
        data = np.concatenate(buffer, axis=0)
        if len(data) < 8000: return # < 0.5s

        try:
            if self.on_status: self.on_status("正在翻译...")
            segments, info = self.model.transcribe(data, beam_size=5)
            text = " ".join([s.text for s in segments]).strip()
            
            if text:
                if self.on_subtitle: self.on_subtitle(text, is_translation=False)
                
                translated = self.translator.translate(text)
                if self.on_subtitle: self.on_subtitle(translated, is_translation=True)
                
        except Exception as e:
            print(f"Transcribe error: {e}")
