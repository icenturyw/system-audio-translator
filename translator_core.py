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
import torch
import httpx

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

class LMStudioTranslator:
    def __init__(self, base_url, model, target_lang="Chinese"):
        self.base_url = base_url.rstrip('/') + "/v1/chat/completions"
        self.model = model
        self.target_lang = target_lang
        self.client = httpx.Client(timeout=10.0)

    def translate(self, text):
        try:
            system_prompt = f"You are a professional translator. Translate the following text into {self.target_lang}. Only return the translated text, no explanations."
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3,
                "stream": False
            }
            
            response = self.client.post(self.base_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"LM Studio Error: {e}")
            return f"[Error: {e}]"

class TranslatorEngine:
    def __init__(self, model_size="tiny", device="auto", compute_type="int8", 
                 source_type="system", target_lang="zh-CN", 
                 api_config=None,
                 on_subtitle=None, on_status=None):
        
        self.on_subtitle = on_subtitle # Callback(text, is_translation, is_final)
        self.on_status = on_status     # Callback(status_text)
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.source_type = source_type
        self.target_lang = target_lang
        self.api_config = api_config or {"type": "google"}
        
        self.running = False
        self.thread = None
        self.q = queue.Queue()
        self.model = None
        self.vad_model = None
        self.translator = None

    def initialize_model(self):
        if self.on_status: self.on_status(f"正在加载模型 ({self.model_size})...")
        
        try:
            # 1. Load Whisper
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

            # 2. Load Silero VAD (Auto-download)
            if self.on_status: self.on_status("正在加载 VAD 神经网络...")
            try:
                # Use local torch hub cache if available, else download
                self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                      model='silero_vad',
                                                      force_reload=False,
                                                      onnx=False) # Use PyTorch version
                self.get_speech_timestamps, _, self.read_audio, _, _ = utils
            except Exception as e:
                print(f"VAD Load Error: {e}")
                # Fallback? No, just fail for now or handle gracefully
                # If offline, this might fail.
                pass

            # 3. Initialize Translator
            if self.api_config.get("type") == "lm_studio":
                self.translator = LMStudioTranslator(
                    base_url=self.api_config.get("url", "http://localhost:1234"),
                    model=self.api_config.get("model", "local-model"),
                    target_lang=self.target_lang
                )
            else:
                self.translator = GoogleTranslator(source='auto', target=self.target_lang)

            if self.on_status: self.on_status(f"就绪 ({self.device})")
            return True
        except Exception as e:
            if self.on_status: self.on_status(f"初始化失败: {e}")
            print(e)
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
            # Device Config (Same as before)
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

            # VAD uses 512, 1024, 1536 samples for 16kHz
            # But the specific model version might be strict about 512.
            # We will read chunks and buffer them to feed exactly 512 to VAD.
            
            stream = p.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=4096, # Read larger chunks for efficiency
                            stream_callback=self._audio_callback)
            
            stream.start_stream()
            
            audio_buffer = []
            vad_accum_buffer = [] # Buffer for VAD processing (float list)
            silence_frames = 0
            is_speaking = False
            
            # VAD Params
            VAD_WINDOW = 512 # Silero strict requirement for 16k
            VAD_THRESHOLD = 0.5 
            MAX_DURATION = 10.0 
            MIN_SILENCE_DURATION = 0.5 
            
            # Streaming Params
            last_intermediate_time = time.time()
            INTERMEDIATE_INTERVAL = 0.8 

            # How many 512-chunks per second? ~31.25
            min_silence_chunks = int(MIN_SILENCE_DURATION * (16000 / VAD_WINDOW))
            
            while self.running:
                try:
                    in_data = self.q.get(timeout=1)
                except queue.Empty:
                    continue

                # 1. Preprocess Audio
                audio_chunk = np.frombuffer(in_data, dtype=np.int16)
                audio_float = audio_chunk.astype(np.float32) / 32768.0

                if channels > 1:
                    audio_float = audio_float.reshape(-1, channels)
                    audio_float = np.mean(audio_float, axis=1)

                if rate != 16000:
                    num_samples = int(len(audio_float) * 16000 / rate)
                    audio_float = signal.resample(audio_float, num_samples)

                # Add to VAD buffer
                vad_accum_buffer.extend(audio_float.tolist())
                
                # Process in 512-sample chunks
                while len(vad_accum_buffer) >= VAD_WINDOW:
                    # Pop 512 samples
                    vad_chunk = np.array(vad_accum_buffer[:VAD_WINDOW], dtype=np.float32)
                    vad_accum_buffer = vad_accum_buffer[VAD_WINDOW:]
                    
                    # 2. VAD Check
                    if self.vad_model:
                        # Reset states if needed? Silero VAD is usually stateless between calls unless using LSTM variant with states
                        # The default 'silero_vad' loaded via hub is usually the standard one.
                        tensor = torch.from_numpy(vad_chunk).unsqueeze(0)
                        try:
                            speech_prob = self.vad_model(tensor, 16000).item()
                            is_speech_frame = speech_prob > VAD_THRESHOLD
                        except:
                            is_speech_frame = False # Fallback
                    else:
                        rms = np.sqrt(np.mean(vad_chunk**2))
                        is_speech_frame = rms > 0.01

                    # 3. Logic State Machine (Per VAD chunk)
                    if is_speech_frame:
                        if not is_speaking:
                            is_speaking = True
                            if self.on_status: self.on_status("正在收听...")
                        
                        silence_frames = 0
                        audio_buffer.extend(vad_chunk.tolist())
                    else:
                        if is_speaking:
                            silence_frames += 1
                            audio_buffer.extend(vad_chunk.tolist())
                            
                            if silence_frames > min_silence_chunks:
                                # Finalize
                                self._transcribe(np.array(audio_buffer, dtype=np.float32), is_final=True)
                                audio_buffer = []
                                is_speaking = False
                                silence_frames = 0
                                if self.on_status: self.on_status("等待语音...")
                
                # 4. Intermediate Transcription (Time-based check outside the VAD loop)
                if is_speaking:
                    current_time = time.time()
                    if current_time - last_intermediate_time > INTERMEDIATE_INTERVAL:
                        # Check buffer length (at least 1s)
                        if len(audio_buffer) > 16000: 
                            self._transcribe(np.array(audio_buffer, dtype=np.float32), is_final=False)
                            last_intermediate_time = current_time
                            
                    # 5. Force Cut
                    current_buffer_duration = len(audio_buffer) / 16000.0
                    if current_buffer_duration > MAX_DURATION:
                         if self.on_status: self.on_status("强制断句...")
                         self._transcribe(np.array(audio_buffer, dtype=np.float32), is_final=True)
                         audio_buffer = []
                         silence_frames = 0
                         last_intermediate_time = time.time()
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            if self.on_status: self.on_status(f"音频错误: {e}")
            import traceback
            traceback.print_exc()

    def _transcribe(self, buffer, is_final=True):
        if len(buffer) == 0: return
        data = buffer # It's already concatenated numpy array in the new logic
        
        # Don't transcribe extremely short noises (< 0.3s) even if VAD triggered
        if len(data) < 16000 * 0.3: return 

        try:
            # For intermediate, we might want faster beam size or greedy
            beam = 5 if is_final else 1 
            
            segments, info = self.model.transcribe(data, beam_size=beam)
            text = " ".join([s.text for s in segments]).strip()
            
            if text:
                # Callback to GUI
                # If it's intermediate, we ONLY show original text, don't translate yet (save API calls)
                # Or translate if you are rich. Let's translate only on final for speed & cost.
                
                if is_final:
                    if self.on_subtitle: self.on_subtitle(text, is_translation=False, is_final=True)
                    translated = self.translator.translate(text)
                    if self.on_subtitle: self.on_subtitle(translated, is_translation=True, is_final=True)
                else:
                    # Intermediate: Only update original text view, mark as not final
                    if self.on_subtitle: self.on_subtitle(text, is_translation=False, is_final=False)
                
        except Exception as e:
            print(f"Transcribe error: {e}")