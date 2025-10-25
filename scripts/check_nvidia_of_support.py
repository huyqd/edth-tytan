"""
Check if NVIDIA Hardware Optical Flow is supported on this system.

This script verifies:
1. NVIDIA GPU present
2. GPU has compute capability >= 7.5 (Turing or newer)
3. NVIDIA Optical Flow SDK installed
4. Can initialize hardware optical flow

Usage:
    python scripts/check_nvidia_of_support.py
"""

import subprocess
import sys

def check_nvidia_smi():
    """Check if nvidia-smi is available and get GPU info."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, None


def check_nvidia_of_sdk():
    """Check if NVIDIA Optical Flow SDK is installed."""
    try:
        from NvOFCuda import NvOFCuda
        return True, "NvOFCuda module found"
    except ImportError as e:
        return False, str(e)


def main():
    print("="*70)
    print("NVIDIA HARDWARE OPTICAL FLOW - COMPATIBILITY CHECK")
    print("="*70)

    all_checks_passed = True

    # Check 1: NVIDIA GPU
    print("\n[1/3] Checking for NVIDIA GPU...")
    nvidia_present, gpu_info = check_nvidia_smi()

    if nvidia_present:
        gpu_name, compute_cap = gpu_info.rsplit(',', 1)
        gpu_name = gpu_name.strip()
        compute_cap = float(compute_cap.strip())

        print(f"  ✅ GPU found: {gpu_name}")
        print(f"  ✅ Compute capability: {compute_cap}")

        # Check 2: Compute capability
        print("\n[2/3] Checking compute capability...")
        if compute_cap >= 7.5:
            print(f"  ✅ Compute capability {compute_cap} >= 7.5 (Turing or newer)")
            print("  ✅ Hardware Optical Flow SUPPORTED")
        else:
            print(f"  ❌ Compute capability {compute_cap} < 7.5")
            print("  ❌ Hardware Optical Flow NOT supported")
            print("\n  Supported GPUs:")
            print("    - RTX 20xx series (Turing, compute 7.5)")
            print("    - RTX 30xx series (Ampere, compute 8.6)")
            print("    - RTX 40xx series (Ada, compute 8.9)")
            print("    - Jetson Orin (Ampere, compute 8.7)")
            all_checks_passed = False
    else:
        print("  ❌ No NVIDIA GPU found")
        print("  ❌ nvidia-smi not available")
        all_checks_passed = False

    # Check 3: NVIDIA OF SDK
    print("\n[3/3] Checking NVIDIA Optical Flow SDK...")
    sdk_installed, sdk_message = check_nvidia_of_sdk()

    if sdk_installed:
        print(f"  ✅ {sdk_message}")

        # Try to initialize
        try:
            from NvOFCuda import NvOFCuda
            nv_of = NvOFCuda()
            print("  ✅ Successfully initialized NvOFCuda")

            # Test with dummy data
            import numpy as np
            dummy1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            dummy2 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

            flow = nv_of.compute_flow(dummy1, dummy2, grid_size=1)
            print(f"  ✅ Test computation successful (flow shape: {flow.shape})")

        except Exception as e:
            print(f"  ❌ Failed to initialize: {e}")
            all_checks_passed = False

    else:
        print(f"  ❌ NVIDIA Optical Flow SDK not installed")
        print(f"     Error: {sdk_message}")
        print("\n  To install:")
        print("    git clone https://github.com/NVIDIA/NVIDIAOpticalFlowSDK.git")
        print("    cd NVIDIAOpticalFlowSDK/NvOFPy")
        print("    python setup.py install")
        all_checks_passed = False

    # Summary
    print("\n" + "="*70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nYou can use NVIDIA Hardware Optical Flow!")
        print("\nNext steps:")
        print("  1. Preprocess test split:")
        print("     python scripts/preprocess_test_split.py")
        print("\n  2. Run inference with NVIDIA HW flow:")
        print("     python scripts/inference_nvidia_flow.py --split-set test")
        return_code = 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nNVIDIA Hardware Optical Flow not available on this system.")
        print("You can still use classical Lucas-Kanade optical flow.")
        return_code = 1

    print("="*70)
    sys.exit(return_code)


if __name__ == '__main__':
    main()
