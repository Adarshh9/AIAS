#!/usr/bin/env python3
"""
AIAS - AI Assistant System
Main entry point
"""

import sys
import os
import argparse
from pathlib import Path

# Load .env file if exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from aias.logger import setup_logging, logger
from aias.orchestrator import AIASOrchestrator, AIASLite
from aias.utils import (
    check_gpu_availability,
    check_audio_devices,
    check_dependencies,
    get_system_info
)


def print_banner():
    """Print AIAS banner"""
    banner = """
    ╔═════════════════════════════════════════════════════╗
    ║                                                     ║
    ║     █████╗ ██╗ █████╗ ███████╗                      ║
    ║    ██╔══██╗██║██╔══██╗██╔════╝                      ║
    ║    ███████║██║███████║███████╗                      ║
    ║    ██╔══██║██║██╔══██║╚════██║                      ║
    ║    ██║  ██║██║██║  ██║███████║                      ║
    ║    ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝                      ║
    ║                                                     ║
    ║    AI Assistant System                              ║
    ║    24/7 Screen-Aware Voice Assistant                ║
    ║                                                     ║
    ╚═════════════════════════════════════════════════════╝
    """
    print(banner)


def run_diagnostics():
    """Run system diagnostics"""
    print("\n📊 AIAS System Diagnostics\n")
    print("=" * 50)
    
    # System info
    print("\n🖥️  System Information:")
    sys_info = get_system_info()
    print(f"   Platform: {sys_info['platform']}")
    print(f"   Python: {sys_info['python_version'].split()[0]}")
    print(f"   CPU Cores: {sys_info['cpu_count']}")
    print(f"   RAM: {sys_info['ram_gb']}GB (Available: {sys_info['ram_available_gb']}GB)")
    
    # GPU info
    print("\n🎮 GPU Information:")
    gpu_info = check_gpu_availability()
    if gpu_info["cuda_available"]:
        print(f"   CUDA Available: ✅")
        print(f"   GPU Count: {gpu_info['gpu_count']}")
        for gpu in gpu_info["gpus"]:
            print(f"   • {gpu['name']}: {gpu['total_vram_gb']}GB VRAM")
        print(f"   Recommended Model: {gpu_info['recommended_model']}")
        if "warning" in gpu_info:
            print(f"   ⚠️  {gpu_info['warning']}")
    else:
        print(f"   CUDA Available: ❌")
        print(f"   Note: CPU inference will be very slow")
    
    # Audio devices
    print("\n🎤 Audio Devices:")
    audio_info = check_audio_devices()
    if audio_info["available"]:
        print(f"   Audio Available: ✅")
        print(f"   Input Devices: {len(audio_info['devices'])}")
        if audio_info["default_device"]:
            print(f"   Default: {audio_info['default_device']['name']}")
        for dev in audio_info["devices"][:3]:  # Show first 3
            print(f"   • {dev['name']}")
    else:
        print(f"   Audio Available: ❌")
        print(f"   Error: {audio_info.get('error', 'Unknown')}")
    
    # Dependencies
    print("\n📦 Dependencies:")
    deps = check_dependencies()
    missing = []
    for pkg, available in deps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {pkg}")
        if not available:
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
    
    print("\n" + "=" * 50)
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if not gpu_info["cuda_available"]:
        print("   • Install CUDA and PyTorch with GPU support for better performance")
    
    if "torch" not in deps or not deps["torch"]:
        print("   • Install PyTorch: pip install torch")
    
    if "pvporcupine" not in deps or not deps["pvporcupine"]:
        print("   • Install Porcupine for hotword detection: pip install pvporcupine")
        print("     (Get access key at https://picovoice.ai/)")
    
    if missing:
        print(f"   • Install missing packages to enable all features")
    
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AIAS - AI Assistant System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with hotword detection
  python main.py --lite             # Start without audio (keyboard input)
  python main.py --diagnostics      # Run system diagnostics
  python main.py -c custom.yaml     # Use custom config file
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        help="Path to config.yaml",
        default=None
    )
    
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Run in lite mode (no audio pipeline, keyboard input only)"
    )
    
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run system diagnostics and exit"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Don't show banner on startup"
    )
    
    args = parser.parse_args()
    
    # Setup loguru logging
    run_dir = setup_logging("logs")
    logger.info(f"AIAS starting - logs at {run_dir}")
    
    # Show banner
    if not args.no_banner:
        print_banner()
    
    # Run diagnostics
    if args.diagnostics:
        run_diagnostics()
        return 0
    
    # Start AIAS
    try:
        if args.lite:
            print("Starting AIAS in Lite mode (keyboard input)...")
            aias = AIASLite(config_path=args.config)
            aias.start()
            
            print("\nAIAS Lite ready. Type your questions below.")
            print("Commands: 'quit' to exit, 'status' for status\n")
            
            while True:
                try:
                    query = input("You: ").strip()
                    
                    if query.lower() in ('quit', 'exit', 'q'):
                        break
                    elif query.lower() == 'status':
                        print(f"Running: {aias._running}")
                        print(f"Screen buffer: {aias._screen.buffer_size if aias._screen else 0}")
                        continue
                    elif not query:
                        continue
                    
                    response = aias.query(query)
                    print(f"\nAIAS: {response}\n")
                    
                except EOFError:
                    break
            
            aias.stop()
            
        else:
            print("Starting AIAS with full audio pipeline...")
            print("Say the wake word followed by your question.\n")
            
            aias = AIASOrchestrator(config_path=args.config)
            aias.run_forever()
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    print("AIAS stopped. Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
