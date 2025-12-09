import customtkinter as ctk
import threading
from translator_core import TranslatorEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI 实时同声传译")
        self.geometry("900x600")
        
        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Sidebar (Controls)
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="🎙️ AI Translator", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Model Selection
        self.model_label = ctk.CTkLabel(self.sidebar, text="模型大小:", anchor="w")
        self.model_label.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.model_option = ctk.CTkOptionMenu(self.sidebar, values=["tiny", "base", "small", "medium", "large-v3"])
        self.model_option.grid(row=2, column=0, padx=20, pady=(0, 10))
        self.model_option.set("medium") # Default

        # Source Selection
        self.source_label = ctk.CTkLabel(self.sidebar, text="输入源:", anchor="w")
        self.source_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.source_option = ctk.CTkOptionMenu(self.sidebar, values=["System Audio (系统)", "Microphone (麦克风)"])
        self.source_option.grid(row=4, column=0, padx=20, pady=(0, 10))

        # Start/Stop Button
        self.start_button = ctk.CTkButton(self.sidebar, text="启动监听", command=self.toggle_listening)
        self.start_button.grid(row=5, column=0, padx=20, pady=20)

        # Status Label
        self.status_label = ctk.CTkLabel(self.sidebar, text="状态: 就绪", text_color="gray")
        self.status_label.grid(row=7, column=0, padx=20, pady=20)

        # Right Main Area (Subtitles)
        self.main_area = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_rowconfigure(2, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # Current Subtitle Display
        self.current_frame = ctk.CTkFrame(self.main_area, corner_radius=15)
        self.current_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        self.origin_text_label = ctk.CTkLabel(self.current_frame, text="等待语音输入...", 
                                              font=ctk.CTkFont(size=16), text_color="gray", wraplength=600)
        self.origin_text_label.pack(padx=20, pady=(20, 5), anchor="w")

        self.trans_text_label = ctk.CTkLabel(self.current_frame, text="", 
                                             font=ctk.CTkFont(size=28, weight="bold"), text_color="#4CC9F0", wraplength=600)
        self.trans_text_label.pack(padx=20, pady=(5, 20), anchor="w")

        # History Textbox
        self.history_label = ctk.CTkLabel(self.main_area, text="历史记录", anchor="w")
        self.history_label.grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        self.history_box = ctk.CTkTextbox(self.main_area, font=ctk.CTkFont(size=14))
        self.history_box.grid(row=2, column=0, sticky="nsew")

        # Logic
        self.translator = None
        self.is_running = False

    def toggle_listening(self):
        if not self.is_running:
            # Start
            self.is_running = True
            self.start_button.configure(text="停止监听", fg_color="#FF5A5F", hover_color="#C0392B")
            self.model_option.configure(state="disabled")
            self.source_option.configure(state="disabled")
            
            # Get settings
            model = self.model_option.get()
            source = "system" if "System" in self.source_option.get() else "mic"
            
            # Init engine in background
            threading.Thread(target=self.start_engine, args=(model, source), daemon=True).start()
        else:
            # Stop
            self.is_running = False
            self.start_button.configure(text="启动监听", fg_color="#3B8ED0", hover_color="#36719F")
            self.model_option.configure(state="normal")
            self.source_option.configure(state="normal")
            
            if self.translator:
                self.translator.stop()
                self.translator = None

    def start_engine(self, model, source):
        self.translator = TranslatorEngine(
            model_size=model,
            source_type=source,
            on_subtitle=self.update_subtitle,
            on_status=self.update_status
        )
        self.translator.start()

    def update_status(self, text):
        self.status_label.configure(text=f"状态: {text}")

    def update_subtitle(self, text, is_translation):
        # This method is called from background thread, so be careful with UI updates?
        # CustomTkinter usually handles thread safety better than vanilla tkinter, but .after is safest.
        self.after(0, lambda: self._update_ui(text, is_translation))

    def _update_ui(self, text, is_translation):
        if is_translation:
            self.trans_text_label.configure(text=text)
            # Add to history
            self.history_box.insert("end", f"译文: {text}\n\n")
            self.history_box.see("end")
        else:
            self.origin_text_label.configure(text=text)
            self.history_box.insert("end", f"原文: {text}\n")

if __name__ == "__main__":
    app = App()
    app.mainloop()
