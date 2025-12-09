import PyInstaller.__main__
import customtkinter
import os
import site
import sys

# 1. 找到 CustomTkinter 路径
ctk_path = os.path.dirname(customtkinter.__file__)
print(f"CustomTkinter path: {ctk_path}")

# 2. 找到 NVIDIA DLL 路径
# 我们需要把 cublas 和 cudnn 的 bin 目录下的 .dll 文件打包进去
nvidia_data = []

try:
    import nvidia.cublas
    import nvidia.cudnn
    
    # 再次修正：使用 __path__ 列表获取路径 (针对 namespace package)
    cublas_root = list(nvidia.cublas.__path__)[0]
    cudnn_root = list(nvidia.cudnn.__path__)[0]
    
    cublas_path = os.path.join(cublas_root, 'bin')
    cudnn_path = os.path.join(cudnn_root, 'bin')
    
    print(f"cuBLAS path: {cublas_path}")
    print(f"cuDNN path: {cudnn_path}")
    
    # Format: "source_path;dest_path" (Windows uses ;)
    nvidia_data.append(f"{cublas_path};nvidia/cublas/bin")
    nvidia_data.append(f"{cudnn_path};nvidia/cudnn/bin")
    
    # Add lib path too just in case (translator_core logic looks for it)
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
        cublas_lib_path = os.path.dirname(nvidia.cublas.lib.__file__)
        cudnn_lib_path = os.path.dirname(nvidia.cudnn.lib.__file__)
        nvidia_data.append(f"{cublas_lib_path};nvidia/cublas/lib")
        nvidia_data.append(f"{cudnn_lib_path};nvidia/cudnn/lib")
    except:
        pass
        
except ImportError as e:
    print(f"Warning: Could not find NVIDIA libraries automatically. {e}")
    # Fallback: Try to find in venv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_site = os.path.join(base_dir, 'venv', 'Lib', 'site-packages')
    if os.path.exists(os.path.join(venv_site, 'nvidia')):
         nvidia_data.append(f"{os.path.join(venv_site, 'nvidia', 'cublas', 'bin')};nvidia/cublas/bin")
         nvidia_data.append(f"{os.path.join(venv_site, 'nvidia', 'cudnn', 'bin')};nvidia/cudnn/bin")

# Build arguments
args = [
    'gui.py',  # 主程序入口
    '--name=AI_Translator',  # exe 名字
    '--onedir',  # 文件夹模式 (推荐，因为 CUDA 库很大，单文件启动会解压很久)
    '--windowed',  # 不显示控制台窗口 (黑框)
    '--noconfirm',  # 覆盖输出目录不询问
    '--clean',  # 清理缓存
    
    # 添加 CustomTkinter 资源
    f'--add-data={ctk_path};customtkinter/',
    
    # 排除不需要的大包 (虽然 PyInstaller 会自动分析，但显式排除更安全)
    # '--exclude-module=matplotlib', 
]

# 添加 NVIDIA 库
for data in nvidia_data:
    args.append(f'--add-data={data}')

print("开始打包...")
print(f"Arguments: {args}")

PyInstaller.__main__.run(args)

print("\n打包完成！")
print("请在 'dist/AI_Translator' 文件夹中找到 'AI_Translator.exe' 并运行测试。")
