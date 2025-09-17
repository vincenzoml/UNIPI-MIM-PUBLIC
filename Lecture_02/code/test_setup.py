#!/usr/bin/env python3
"""
SimpleITK Setup Test - Lecture 02
Quick test to verify SimpleITK installation and basic functionality.
"""

def test_imports():
    """Test if all required libraries can be imported."""
    print("Testing imports...")
    
    try:
        import SimpleITK as sitk
        print("✓ SimpleITK imported successfully")
        print(f"  Version: {sitk.Version()}")
    except ImportError:
        print("✗ SimpleITK import failed")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  Version: {np.__version__}")
    except ImportError:
        print("✗ NumPy import failed")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
        import matplotlib
        print(f"  Version: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib import failed")
        return False
    
    return True


def test_basic_functionality():
    """Test basic SimpleITK operations."""
    print("\nTesting basic SimpleITK functionality...")
    
    import SimpleITK as sitk
    import numpy as np
    
    # Create a simple 2D image
    array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    image = sitk.GetImageFromArray(array)
    
    print(f"✓ Created image with size: {image.GetSize()}")
    
    # Apply a simple filter
    smoothed = sitk.SmoothingRecursiveGaussian(image, sigma=1.0)
    print(f"✓ Applied Gaussian smoothing")
    
    # Convert back to numpy
    result_array = sitk.GetArrayFromImage(smoothed)
    print(f"✓ Converted back to NumPy array: {result_array.shape}")
    
    return True


def main():
    """Main test function."""
    print("=" * 50)
    print("SimpleITK Setup Test - Lecture 02")
    print("=" * 50)
    
    if not test_imports():
        print("\n❌ Setup incomplete. Please install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Setup is complete.")
    print("You can now run: python simpleitk_introduction.py")
    print("=" * 50)
    return True


if __name__ == "__main__":
    main()