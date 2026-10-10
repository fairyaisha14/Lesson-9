
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Load the original image
# ----------------------------------

image = cv2.imread("family.webp")

if image is None:
    print("Error: Could not find family.webp.")
    print("Make sure the image is in the same folder as main.py.")
    exit()

# Convert BGR to RGB for correct display in Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ----------------------------------
# 2. Apply image filtering
# ----------------------------------

# Gaussian filter reduces noise and smooths the image
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# Median filter helps remove salt-and-pepper noise
median = cv2.medianBlur(gray, 5)

# Apply a larger Gaussian blur for comparison
gaussian_strong = cv2.GaussianBlur(gray, (9, 9), 0)


# ----------------------------------
# 3. Sobel edge detection
# ----------------------------------

# Detect horizontal intensity changes
sobel_x = cv2.Sobel(
    gray, cv2.CV_64F, 1, 0, ksize=3
)

# Detect vertical intensity changes
sobel_y = cv2.Sobel(
    gray, cv2.CV_64F, 0, 1, ksize=3
)

# Combine horizontal and vertical edges
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Convert the result to an 8-bit image
sobel = cv2.convertScaleAbs(sobel_magnitude)


# ----------------------------------
# 4. Canny edge detection
# ----------------------------------

# Change these values to experiment
lower_threshold = 100
upper_threshold = 200

canny = cv2.Canny(
    gray, lower_threshold, upper_threshold
)

# Canny on the Gaussian-filtered image
canny_filtered = cv2.Canny(
    gaussian, lower_threshold, upper_threshold
)


# ----------------------------------
# 5. Laplacian edge detection
# ----------------------------------

laplacian_result = cv2.Laplacian(
    gray, cv2.CV_64F, ksize=3
)

laplacian = cv2.convertScaleAbs(laplacian_result)


# ----------------------------------
# 6. Edge detection after filtering
# ----------------------------------

# Sobel on the Gaussian-filtered image
sobel_filtered_x = cv2.Sobel(
    gaussian, cv2.CV_64F, 1, 0, ksize=3
)

sobel_filtered_y = cv2.Sobel(
    gaussian, cv2.CV_64F, 0, 1, ksize=3
)

sobel_filtered = cv2.convertScaleAbs(
    cv2.magnitude(sobel_filtered_x, sobel_filtered_y)
)

# Laplacian on the Median-filtered image
laplacian_median = cv2.Laplacian(
    median, cv2.CV_64F, ksize=3
)

laplacian_median = cv2.convertScaleAbs(
    laplacian_median
)


# ----------------------------------
# 7. Display the results
# ----------------------------------

images = [
    image_rgb,
    gray,
    gaussian,
    median,
    gaussian_strong,
    sobel,
    canny,
    laplacian,
    canny_filtered,
    sobel_filtered,
    laplacian_median
]

titles = [
    "Original Family Image",
    "Grayscale Image",
    "Gaussian Filter (5x5)",
    "Median Filter (5x5)",
    "Gaussian Filter (9x9)",
    "Sobel Edge Detection",
    "Canny Edge Detection",
    "Laplacian Edge Detection",
    "Canny After Gaussian",
    "Sobel After Gaussian",
    "Laplacian After Median"
]

plt.figure(figsize=(15, 12))

for i, result in enumerate(images):
    plt.subplot(4, 3, i + 1)

    if i == 0:
        plt.imshow(result)
    else:
        plt.imshow(result, cmap="gray")

    plt.title(titles[i])
    plt.axis("off")

plt.suptitle(
    "Enhancing Edge Detection and Filtering",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ----------------------------------
# 8. Experiment with Canny thresholds
# ----------------------------------

print("\nCanny Threshold Experiment")
print("Lower threshold:", lower_threshold)
print("Upper threshold:", upper_threshold)
print("Change these values in the code to compare results.")

print("\nImage processing completed successfully!")
print("Original image size:", image.shape)
print("Grayscale image size:", gray.shape)
print("All filters and edge detection methods were applied.")
