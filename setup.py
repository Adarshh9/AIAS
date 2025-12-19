#!/usr/bin/env python3
"""
AIAS Quick Start Script
Sets up environment and verifies installation
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display result"""
    print(f"\n📌 {description}")
    print(f"   Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ Success")
        return True
    else:
        print(f"   ❌ Failed")
        if result.stderr:
            print(f"   Error: {result.stderr[:200]}")
        return False


def main():
    print("=" * 60)
    print("🚀 AIAS Quick Setup")
    print("=" * 60)
    
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"\n📁 Project directory: {project_dir}")
    
    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("   ⚠️  Python 3.10+ recommended")
    
    # Check if venv exists
    venv_path = project_dir / "venv"
    if not venv_path.exists():
        print("\n📦 Creating virtual environment...")
        run_command("python -m venv venv", "Creating venv")
    else:
        print("\n✅ Virtual environment already exists")
    
    # Determine pip path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Install/upgrade pip
    run_command(f'"{python_path}" -m pip install --upgrade pip', "Upgrading pip")
    
    # Install PyTorch with CUDA
    print("\n🔥 Installing PyTorch with CUDA support...")
    pytorch_cmd = f'"{pip_path}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'
    run_command(pytorch_cmd, "Installing PyTorch")
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    run_command(f'"{pip_path}" install -r requirements.txt', "Installing requirements")
    
    # Install additional packages that may be missing
    extra_packages = [
        "faster-whisper",
        "pyaudio",
        "mss",
        "pillow",
        "pyyaml",
        "psutil",
        "transformers>=4.40.0",
        "accelerate",
        "einops"
    ]
    
    for pkg in extra_packages:
        run_command(f'"{pip_path}" install {pkg}', f"Installing {pkg}")
    
    # Verify installation
    print("\n\n" + "=" * 60)
    print("🔍 Verifying Installation")
    print("=" * 60)
    
    verify_script = '''
import sys
print("Checking imports...")

modules = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("PIL", "Pillow"),
    ("mss", "MSS (Screenshots)"),
    ("yaml", "PyYAML"),
    ("numpy", "NumPy"),
]

failed = []
for module, name in modules:
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError as e:
        print(f"  ❌ {name}: {e}")
        failed.append(name)

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠️  CUDA not available (CPU only)")
except:
    pass

# Optional modules
optional = [
    ("pyaudio", "PyAudio"),
    ("faster_whisper", "Faster-Whisper"),
    ("pvporcupine", "Porcupine (hotword)"),
    ("webrtcvad", "WebRTC VAD"),
]

print("\\nOptional modules:")
for module, name in optional:
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ⚪ {name} (not installed)")

if failed:
    print(f"\\n❌ {len(failed)} required modules failed to import")
    sys.exit(1)
else:
    print("\\n✅ All required modules imported successfully!")
'''
    
    verify_result = subprocess.run(
        [str(python_path), "-c", verify_script],
        capture_output=True,
        text=True
    )
    print(verify_result.stdout)
    if verify_result.stderr:
        print(verify_result.stderr)
    
    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    
    print("""
Next steps:

1. Activate the virtual environment:
   Windows:  .\\venv\\Scripts\\activate
   Linux:    source venv/bin/activate

2. Run diagnostics:
   python main.py --diagnostics

3. Start AIAS:
   python main.py           # Full mode (with hotword)
   python main.py --lite    # Lite mode (keyboard input)

4. (Optional) Set Picovoice access key for custom wake words:
   set PICOVOICE_ACCESS_KEY=your_key_here

For troubleshooting, see README.md
""")


if __name__ == "__main__":
    main()
