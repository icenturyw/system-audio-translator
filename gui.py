import customtkinter as ctk
import threading
import json
import os
from translator_core import TranslatorEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "settings.json"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI 实时同声传译")
        self.geometry("1000x650")
        
        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Logic Vars
        self.translator = None
        self.is_running = False
        
        # Load Config
        self.config = self.load_config()

        # --- Left Sidebar (Controls) ---
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(8, weight=1)

        # Logo
        self.logo_label = ctk.CTkLabel(self.sidebar, text="🎙️ AI Translator", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # 1. Model Selection
        self.model_label = ctk.CTkLabel(self.sidebar, text="模型大小 (Model):", anchor="w")
        self.model_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.model_option = ctk.CTkOptionMenu(self.sidebar, values=["tiny", "base", "small", "medium", "large-v3"])
        self.model_option.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.model_option.set(self.config.get("model", "small"))

        # 2. Source Selection
        self.source_label = ctk.CTkLabel(self.sidebar, text="输入源 (Input):", anchor="w")
        self.source_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.source_option = ctk.CTkOptionMenu(self.sidebar, values=["System Audio (系统)", "Microphone (麦克风)"])
        self.source_option.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.source_option.set(self.config.get("source", "System Audio (系统)"))

        # 3. Target Language Selection (NEW)
        self.lang_label = ctk.CTkLabel(self.sidebar, text="目标语言 (Target):", anchor="w")
        self.lang_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        
        # Map display names to codes
        self.lang_map = {
            "Chinese (中文)": "zh-CN",
            "English (英语)": "en",
            "Japanese (日语)": "ja",
            "Korean (韩语)": "ko",
            "French (法语)": "fr",
            "German (德语)": "de",
            "Russian (俄语)": "ru",
            "Spanish (西语)": "es",
            "Hindi (印地语)": "hi"
        }
        self.lang_option = ctk.CTkOptionMenu(self.sidebar, values=list(self.lang_map.keys()))
        self.lang_option.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")
        # Set default from config
        saved_lang_code = self.config.get("target_lang", "zh-CN")
        for name, code in self.lang_map.items():
            if code == saved_lang_code:
                self.lang_option.set(name)
                break

        # 4. Start/Stop Button
        self.start_button = ctk.CTkButton(self.sidebar, text="启动监听", command=self.toggle_listening, height=40, font=ctk.CTkFont(weight="bold"))
        self.start_button.grid(row=7, column=0, padx=20, pady=20, sticky="ew")

        # 5. Always on Top (NEW)
        self.top_switch = ctk.CTkSwitch(self.sidebar, text="窗口置顶", command=self.toggle_topmost)
        self.top_switch.grid(row=9, column=0, padx=20, pady=(0, 10), sticky="w")
        if self.config.get("always_on_top", False):
            self.top_switch.select()
            self.attributes('-topmost', True)

        # 6. Mini Mode Button
        self.mini_btn = ctk.CTkButton(self.sidebar, text="进入精简模式 ↗", command=self.enable_mini_mode, fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        self.mini_btn.grid(row=10, column=0, padx=20, pady=(0, 20), sticky="ew")

        # Status Label
        self.status_label = ctk.CTkLabel(self.sidebar, text="状态: 就绪", text_color="gray", anchor="w")
        self.status_label.grid(row=11, column=0, padx=20, pady=(0, 20), sticky="w")

        # --- Right Main Area (Subtitles) ---
        self.main_area = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_rowconfigure(2, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # Current Subtitle Display
        self.current_frame = ctk.CTkFrame(self.main_area, corner_radius=15, fg_color=("#EBEBEB", "#2B2B2B"))
        self.current_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.current_frame.grid_columnconfigure(0, weight=1) # Make column 0 expandable for text

        # Restore Button (Hidden by default, visible in Mini Mode)
        self.restore_btn = ctk.CTkButton(self.current_frame, text="⬜", width=30, height=30, 
                                         fg_color="transparent", hover_color=("gray70", "gray30"),
                                         command=self.disable_mini_mode)
        # We don't grid it yet, only in mini mode

        self.origin_text_label = ctk.CTkLabel(self.current_frame, text="等待语音输入...", 
                                              font=ctk.CTkFont(size=16), text_color="gray", wraplength=650, anchor="w", justify="left")
        self.origin_text_label.grid(row=0, column=0, padx=25, pady=(20, 5), sticky="ew")

        self.trans_text_label = ctk.CTkLabel(self.current_frame, text="", 
                                             font=ctk.CTkFont(size=26, weight="bold"), text_color="#4CC9F0", wraplength=650, anchor="w", justify="left")
        self.trans_text_label.grid(row=1, column=0, padx=25, pady=(5, 20), sticky="ew")

        # History Textbox
        self.history_label = ctk.CTkLabel(self.main_area, text="历史记录 (History)", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.history_label.grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        self.history_box = ctk.CTkTextbox(self.main_area, font=ctk.CTkFont(size=14), activate_scrollbars=True)
        self.history_box.grid(row=2, column=0, sticky="nsew")

        # Save config on close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def enable_mini_mode(self):
        # 1. Hide non-essential UI
        self.sidebar.grid_remove()
        self.history_label.grid_remove()
        self.history_box.grid_remove()
        
        # 2. Adjust Layout
        self.main_area.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        self.current_frame.grid(pady=0) # Remove padding for tighter fit
        
        # 3. Show Restore Button (Top Right of the card)
        self.restore_btn.grid(row=0, column=1, sticky="ne", padx=10, pady=10)
        
        # 4. Resize Window
        self.geometry("800x200")
        
        # 5. Force Topmost (Optional, but recommended for mini mode)
        self.attributes('-topmost', True)

    def disable_mini_mode(self):
        # 1. Restore UI
        self.sidebar.grid()
        self.history_label.grid()
        self.history_box.grid()
        
        # 2. Restore Layout
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.current_frame.grid(pady=(0, 20))
        
        # 3. Hide Restore Button
        self.restore_btn.grid_remove()
        
        # 4. Restore Window Size
        self.geometry("1000x650")
        
        # 5. Restore Topmost state based on switch
        self.attributes('-topmost', bool(self.top_switch.get()))

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self):
        self.config["model"] = self.model_option.get()
        self.config["source"] = self.source_option.get()
        self.config["target_lang"] = self.lang_map[self.lang_option.get()]
        self.config["always_on_top"] = bool(self.top_switch.get())
        
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def toggle_topmost(self):
        is_top = bool(self.top_switch.get())
        self.attributes('-topmost', is_top)

    def toggle_listening(self):
        if not self.is_running:
            # Start
            self.is_running = True
            self.start_button.configure(text="🛑 停止监听", fg_color="#FF5A5F", hover_color="#C0392B")
            self.disable_controls(True)
            
            # Get settings
            model = self.model_option.get()
            source = "system" if "System" in self.source_option.get() else "mic"
            target_lang = self.lang_map[self.lang_option.get()]
            
            # Init engine in background
            threading.Thread(target=self.start_engine, args=(model, source, target_lang), daemon=True).start()
        else:
            # Stop
            self.is_running = False
            self.start_button.configure(text="启动监听", fg_color="#3B8ED0", hover_color="#36719F")
            self.disable_controls(False)
            
            if self.translator:
                self.translator.stop()
                self.translator = None

    def disable_controls(self, disabled):
        state = "disabled" if disabled else "normal"
        self.model_option.configure(state=state)
        self.source_option.configure(state=state)
        self.lang_option.configure(state=state)

    def start_engine(self, model, source, target_lang):
        self.translator = TranslatorEngine(
            model_size=model,
            source_type=source,
            target_lang=target_lang,
            on_subtitle=self.update_subtitle,
            on_status=self.update_status
        )
        self.translator.start()

    def update_status(self, text):
        self.status_label.configure(text=f"状态: {text}")

    def update_subtitle(self, text, is_translation):
        self.after(0, lambda: self._update_ui(text, is_translation))

    def _update_ui(self, text, is_translation):
        if is_translation:
            self.trans_text_label.configure(text=text)
            self.history_box.insert("end", f" {text}\n\n")
            self.history_box.see("end")
        else:
            self.origin_text_label.configure(text=text)
            # Add timestamp or just text? Just text for now.
            self.history_box.insert("end", f"▶ {text}\n")

    def on_close(self):
        self.save_config()
        if self.translator:
            self.translator.stop()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()