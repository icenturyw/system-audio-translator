# Contributing

Thanks for helping improve System Audio Translator. This project is intentionally practical: compatibility reports, bug reproductions, and small reliability fixes are all valuable.

## Development Setup

```powershell
git clone https://github.com/icenturyw/system-audio-translator.git
cd system-audio-translator

python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg if you want to run the translator locally:

```powershell
winget install Gyan.FFmpeg
```

## Quality Checks

Run the lightweight syntax check before opening a pull request:

```powershell
python -m py_compile gui.py main.py translator_core.py build.py check_hostapis.py debug_audio.py
```

When changing audio behavior, also include:

- Windows version.
- Audio device name.
- Input mode, either system audio or microphone.
- Model size and compute device.
- Whether FFmpeg and CUDA runtime wheels are installed.

## Pull Request Guidelines

- Keep changes focused and explain the user-facing behavior.
- Include screenshots for GUI changes.
- Avoid committing generated files from `build/`, `dist/`, virtual environments, or local settings.
- Mention whether the change was tested with real audio hardware.

## Good First Issues

- Improve README examples for different language pairs.
- Add tests around configuration parsing and provider payloads.
- Improve diagnostics when no WASAPI loopback device is found.
- Add release notes for Windows packaged builds.
