#%%  

import SimpleITK as sitk

img = sitk.ReadImage("three_coloured_items.png")
print(f"Image characteristics: {img}")


# Extract channels
r = sitk.VectorIndexSelectionCast(img, 0)
g = sitk.VectorIndexSelectionCast(img, 1)
b = sitk.VectorIndexSelectionCast(img, 2)

# Weighted sum
gray = 0.299*(sitk.Cast(r, sitk.sitkFloat32)) + 0.587*sitk.Cast(g, sitk.sitkFloat32) + 0.114*sitk.Cast(b, sitk.sitkFloat32)

sitk.WriteImage(sitk.Cast(gray, sitk.sitkUInt8), "gray.png")
