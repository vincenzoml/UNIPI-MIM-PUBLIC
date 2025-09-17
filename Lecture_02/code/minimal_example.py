#!/usr/bin/env python3
"""
Minimal SimpleITK Example - Lecture 02
A concise introduction to SimpleITK basics for beginners.
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def minimal_example():
    """Minimal example demonstrating core SimpleITK concepts."""
    print("Creating a simple 2D image...")
    
    # 1. Create a simple 2D image from a NumPy array
    # Create a 128x128 image with a circle in the center
    size = 128
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    
    # Build a distance-squared scalar field (no mask yet)
    # Each pixel value = x^2 + y^2 relative to the image center.
    # We'll derive the circle purely via thresholding this field.
    radius = 30  # desired circle radius in pixels
    distance_sq = (X**2 + Y**2).astype(np.int32)  # range: 0 .. ~ (64^2+64^2)

    # Use the raw distance-squared field as the image intensities.
    # (Lower values near center, higher outward.)
    image_array = distance_sq  # keep as int32; SimpleITK will handle conversion
    
    # Convert NumPy array to SimpleITK Image
    image = sitk.GetImageFromArray(image_array)
    
    # 2. Set image properties (spacing = 1mm per pixel)
    # image.SetSpacing([1.0, 1.0])
    # image.SetOrigin([0.0, 0.0])
    
    # print(f"Image size: {image.GetSize()}")
    # print(f"Image spacing: {image.GetSpacing()}")
    
    # 3. Apply a threshold to CREATE the circle from the distance field.
    #    Pixels with (x^2 + y^2) <= radius^2 become foreground (inside circle).
    lower_threshold = 0
    upper_threshold = radius**2  # inclusive upper bound for inside region
    thresholded = sitk.BinaryThreshold(
        image,
        lowerThreshold=lower_threshold,
        upperThreshold=upper_threshold,
        insideValue=255,
        outsideValue=0,
    )

    # 4. Get basic statistics
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    print(f"Distance^2 image - Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
    
    stats.Execute(thresholded)
    print(f"Thresholded image - Min: {stats.GetMinimum()}, Max: {stats.GetMaximum()}")
    
    # 5. Convert back to NumPy for visualization
    original_array = sitk.GetArrayFromImage(image)
    thresholded_array = sitk.GetArrayFromImage(thresholded)
    
    # 6. Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.imshow(original_array, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(thresholded_array, cmap='gray')
    ax2.set_title('Thresholded Image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 7. Save the result (convert to UInt8 for PNG compatibility)
    thresholded_uint8 = sitk.Cast(thresholded, sitk.sitkUInt8)
    sitk.WriteImage(thresholded_uint8, "thresholded_circle.png")
    print("Saved thresholded image as 'thresholded_circle.png'")


if __name__ == "__main__":
    print("=" * 40)
    print("Minimal SimpleITK Example")
    print("=" * 40)
    
    try:
        minimal_example()
        print("\n✅ Example completed successfully!")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("Install with: pip install SimpleITK numpy matplotlib")
    except Exception as e:
        print(f"❌ Error: {e}")