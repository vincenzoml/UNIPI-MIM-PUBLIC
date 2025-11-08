# %% 

# CODE REPOSITORY: 

# https://github.com/vincenzoml/UNIPI-MIM-PUBLIC

# https://github.com/vincenzoml/UNIPI-MIM-PUBLIC/blob/main/Lecture_07/maze.png

# Primitives that we use:

# sitk.ReadImage("maze.png")  # load an image from file
# sitk.VectorIndexSelectionCast(img, 0) == 255) # select the red channel (0, 1, 2 for RGB)

# sitk.ConnectedComponent(white)  # find connected components in a binary image
# sitk.LabelToRGB(cc)  # convert a label image to RGB for visualization

# sitk.BinaryDilate(white, [5, 5])  # dilate
# sitk.BinaryErode(dilated, [5, 5])  # erode

# arr_view_2 = sitk.GetArrayViewFromImage(eroded) # zero-copy read-only view of the image as a NumPy array












# %% 

import SimpleITK as sitk

# %% Load image

img = sitk.ReadImage("maze.png")

print(f"Image size: {img.GetSize()}")
print(f"Image spacing: {img.GetSpacing()}") 
print(f"Image origin: {img.GetOrigin()}")
print(f"Image direction: {img.GetDirection()}")

# %% Find e.g. the white region

red_component = sitk.VectorIndexSelectionCast(img, 0) 
green_component = sitk.VectorIndexSelectionCast(img, 1) 
blue_component = sitk.VectorIndexSelectionCast(img, 2) 

white = (red_component == 255) & (green_component == 255) & (blue_component == 255)

sitk.WriteImage(red_component, "red.png")
sitk.WriteImage(green_component, "green.png")
sitk.WriteImage(blue_component, "blue.png")
sitk.WriteImage(white * 255, "white.png")


# %% Find the connected components

cc = sitk.ConnectedComponent(white)

# Get statistics about connected components
cc_stats = sitk.LabelShapeStatisticsImageFilter()
cc_stats.Execute(cc)
print(f"Number of components: {cc_stats.GetNumberOfLabels()}")

# We cannot save this because the possible range of the labels is too large for png files
# sitk.WriteImage(cc, "cc.png")

# Convert to RGB with random colors
cc_rgb = sitk.LabelToRGB(cc)
sitk.WriteImage(cc_rgb, "cc_rgb.png")

dilated = sitk.BinaryDilate(white, [2, 2])
eroded = sitk.BinaryErode(dilated, [2, 2])

sitk.WriteImage(sitk.Cast(dilated, sitk.sitkUInt8) * 255, "dilated.png")
sitk.WriteImage(sitk.Cast(eroded, sitk.sitkUInt8) * 255, "eroded.png")


# %%
# Read pixels efficiently (no copy)
# to keep the view *valid* one needs to keep the original image in memory! 

# Let's e.g. compute the volume of the mask (there are more efficient ways to do this!)

arr_view = sitk.GetArrayViewFromImage(white)

count = 0 

for y in range(arr_view.shape[0]):
    for x in range(arr_view.shape[1]):
        if arr_view[y, x] > 0:
            count += 1

print(f"Volume of the white region: {count} pixels")        

# %%

arr_view_2 = sitk.GetArrayViewFromImage(eroded)

count = 0 

for y in range(arr_view_2.shape[0]):
    for x in range(arr_view_2.shape[1]):
        if arr_view_2[y, x] > 0:
            count += 1

print(f"Volume of the eroded region: {count} pixels")

# %% Inefficient way (copy)
arr = sitk.GetArrayFromImage(white)

count = 0

for y in range(arr.shape[0]):
    for x in range(arr.shape[1]):
        count = count + 1

print(f"Volume of the white region: {count} pixels")