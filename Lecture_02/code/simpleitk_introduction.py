#!/usr/bin/env python3
"""
SimpleITK Introduction Exercise - Lecture 02
Formal and Hybrid Methods for Medical Imaging

This self-contained exercise introduces SimpleITK with a 2D synthetic medical image.
The exercise covers:
- Creating and manipulating 2D images
- Basic image operations (filtering, thresholding)
- Image measurements and analysis
- Visualization with matplotlib

Requirements: SimpleITK, numpy, matplotlib
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_synthetic_medical_image(size=(256, 256)):
    """
    Create a synthetic 2D medical-style image with anatomical-like structures.
    
    Args:
        size (tuple): Image dimensions (width, height)
    
    Returns:
        SimpleITK.Image: Synthetic 2D medical image
    """
    print("Creating synthetic 2D medical image...")
    
    # Create coordinate grids
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    X, Y = np.meshgrid(x, y)
    
    # Create synthetic anatomical structures
    # Background tissue
    background = 50 + 20 * np.random.normal(0, 1, size)
    
    # Circular organ (e.g., cross-section of vessel or organ)
    center_organ = np.exp(-((X-0.2)**2 + (Y+0.1)**2) / 0.1) * 150
    
    # Elongated structure (e.g., bone or vessel)
    bone_structure = np.exp(-((X+0.3)**2/0.05 + (Y-0.2)**2/0.3)) * 200
    
    # Small lesion or abnormality
    lesion = np.exp(-((X-0.4)**2 + (Y-0.3)**2) / 0.02) * 180
    
    # Combine structures
    image_array = background + center_organ + bone_structure + lesion
    
    # Add some noise
    image_array += np.random.normal(0, 10, size)
    
    # Ensure positive values and convert to appropriate data type
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    # Convert numpy array to SimpleITK Image
    sitk_image = sitk.GetImageFromArray(image_array)
    
    # Set realistic medical image spacing (0.5mm per pixel)
    sitk_image.SetSpacing([0.5, 0.5])
    sitk_image.SetOrigin([0.0, 0.0])
    
    return sitk_image


def display_image_info(image, title="Image Information"):
    """Display comprehensive information about a SimpleITK image."""
    print(f"\n=== {title} ===")
    print(f"Dimensions: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")
    print(f"Direction: {image.GetDirection()}")
    print(f"Pixel type: {image.GetPixelIDTypeAsString()}")
    print(f"Number of components per pixel: {image.GetNumberOfComponentsPerPixel()}")
    
    # Get image statistics
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    print(f"Min intensity: {stats_filter.GetMinimum():.2f}")
    print(f"Max intensity: {stats_filter.GetMaximum():.2f}")
    print(f"Mean intensity: {stats_filter.GetMean():.2f}")
    print(f"Standard deviation: {stats_filter.GetSigma():.2f}")


def apply_gaussian_smoothing(image, sigma=1.0):
    """Apply Gaussian smoothing to reduce noise."""
    print(f"\nApplying Gaussian smoothing (sigma={sigma})...")
    
    # Ensure sigma is a sequence matching the image dimension (SimpleITK expects a list)
    if isinstance(sigma, (int, float)):
        sigma_list = [float(sigma)] * image.GetDimension()
    else:
        sigma_list = list(sigma)
        if len(sigma_list) != image.GetDimension():
            raise ValueError(f"sigma must be a float or a sequence of length {image.GetDimension()}")
    
    smoothed = sitk.SmoothingRecursiveGaussian(image, sigma_list)
    return smoothed


def apply_edge_detection(image):
    """Apply edge detection using gradient magnitude."""
    print("\nApplying edge detection...")
    
    # Convert to float for gradient computation
    float_image = sitk.Cast(image, sitk.sitkFloat32)
    
    # Compute gradient magnitude
    edges = sitk.GradientMagnitude(float_image)
    
    # Normalize to 0-255 range
    edges = sitk.RescaleIntensity(edges, 0, 255)
    edges = sitk.Cast(edges, sitk.sitkUInt8)
    
    return edges


def apply_thresholding(image, lower_threshold=100):
    """Apply binary thresholding to segment structures."""
    print(f"\nApplying binary thresholding (threshold={lower_threshold})...")
    
    # Binary thresholding
    binary_mask = sitk.BinaryThreshold(image, 
                                      lowerThreshold=lower_threshold, 
                                      upperThreshold=255, 
                                      insideValue=255, 
                                      outsideValue=0)
    return binary_mask


def analyze_connected_components(binary_image):
    """Analyze connected components in binary image."""
    print("\nAnalyzing connected components...")
    
    # Label connected components
    labeled = sitk.ConnectedComponent(binary_image)
    
    # Get statistics for each component
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(labeled)
    
    print(f"Number of connected components: {label_stats.GetNumberOfLabels()}")
    
    for label in label_stats.GetLabels():
        print(f"  Component {label}:")
        print(f"    Area: {label_stats.GetPhysicalSize(label):.2f} mm²")
        print(f"    Centroid: {label_stats.GetCentroid(label)}")
        print(f"    Bounding box: {label_stats.GetBoundingBox(label)}")
    
    return labeled


def visualize_results(original, smoothed, edges, binary_mask, labeled):
    """Create a comprehensive visualization of all processing steps."""
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SimpleITK 2D Image Processing Pipeline', fontsize=16)
    
    # Convert SimpleITK images to numpy arrays for visualization
    original_array = sitk.GetArrayFromImage(original)
    smoothed_array = sitk.GetArrayFromImage(smoothed)
    edges_array = sitk.GetArrayFromImage(edges)
    binary_array = sitk.GetArrayFromImage(binary_mask)
    labeled_array = sitk.GetArrayFromImage(labeled)
    
    # Original image
    axes[0, 0].imshow(original_array, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Smoothed image
    axes[0, 1].imshow(smoothed_array, cmap='gray')
    axes[0, 1].set_title('Gaussian Smoothed')
    axes[0, 1].axis('off')
    
    # Edge detection
    axes[0, 2].imshow(edges_array, cmap='gray')
    axes[0, 2].set_title('Edge Detection')
    axes[0, 2].axis('off')
    
    # Binary threshold
    axes[1, 0].imshow(binary_array, cmap='gray')
    axes[1, 0].set_title('Binary Threshold')
    axes[1, 0].axis('off')
    
    # Connected components
    axes[1, 1].imshow(labeled_array, cmap='nipy_spectral')
    axes[1, 1].set_title('Connected Components')
    axes[1, 1].axis('off')
    
    # Histogram of original image
    axes[1, 2].hist(original_array.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 2].set_title('Intensity Histogram')
    axes[1, 2].set_xlabel('Pixel Intensity')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def save_images(output_dir, **images):
    """Save processed images to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nSaving images to {output_path}...")
    
    for name, image in images.items():
        filename = output_path / f"{name}.png"
        # Convert to UInt8 for PNG compatibility
        image_uint8 = sitk.Cast(sitk.RescaleIntensity(image, 0, 255), sitk.sitkUInt8)
        sitk.WriteImage(image_uint8, str(filename))
        print(f"  Saved: {filename}")


def main():
    """Main exercise function demonstrating SimpleITK capabilities."""
    print("=" * 60)
    print("SimpleITK Introduction Exercise - Lecture 02")
    print("Formal and Hybrid Methods for Medical Imaging")
    print("=" * 60)
    
    # Step 1: Create synthetic medical image
    original_image = create_synthetic_medical_image()
    display_image_info(original_image, "Original Synthetic Image")
    
    # Step 2: Apply Gaussian smoothing
    smoothed_image = apply_gaussian_smoothing(original_image, sigma=1.5)
    display_image_info(smoothed_image, "Smoothed Image")
    
    # Step 3: Edge detection
    edge_image = apply_edge_detection(smoothed_image)
    
    # Step 4: Binary thresholding
    binary_image = apply_thresholding(original_image, lower_threshold=120)
    
    # Step 5: Connected component analysis
    labeled_image = analyze_connected_components(binary_image)
    
    # Step 6: Visualization
    visualize_results(original_image, smoothed_image, edge_image, 
                     binary_image, labeled_image)
    
    # Step 7: Save results
    save_images("output", 
                original=original_image,
                smoothed=smoothed_image,
                edges=edge_image,
                binary=binary_image,
                labeled=labeled_image)
    
    print("\n" + "=" * 60)
    print("Exercise completed successfully!")
    print("Check the 'output' directory for saved images.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: Missing required library - {e}")
        print("Please install requirements: pip install SimpleITK numpy matplotlib")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your Python environment and try again.")