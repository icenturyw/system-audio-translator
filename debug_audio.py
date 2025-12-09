import sounddevice as sd
import sys

print(f"Python executable: {sys.executable}")
print(f"Sounddevice version: {sd.__version__}")
try:
    print(f"PortAudio version: {sd.get_portaudio_version()}")
except:
    print("Could not get PortAudio version")

print("\nDefault devices:")
print(sd.query_devices())

print("\nTesting InputStream signature...")
import inspect
sig = inspect.signature(sd.InputStream)
print(f"InputStream signature: {sig}")
if 'loopback' in sig.parameters:
    print("SUCCESS: 'loopback' parameter is present.")
else:
    print("FAILURE: 'loopback' parameter is MISSING.")
