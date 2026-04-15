# Input image path
image_path = r"your_image.png"

# Load image
img = load_image(image_path)

# Set parameters
m = 10   # max threshold of reference image
t = 7    # max value of input image
k = 40   # number of neighbors

# Run preprocessing
imgnoword, imgadapt = pre_process_hsv(img, t=t, k=k, m=m)

# Show results like MATLAB montage
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(imgnoword)
axes[1].set_title("After annotation removal")
axes[1].axis("off")

axes[2].imshow(imgadapt)
axes[2].set_title("After threshold adaptation")
axes[2].axis("off")

plt.tight_layout()
plt.show()
