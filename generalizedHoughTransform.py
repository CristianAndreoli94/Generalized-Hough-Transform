import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def buildingReferenceTable(template):
    """
    Building the reference table for the input image.
    The reference table is a dictionary, using theta as the key, that stores the (r, alpha) pairs for each edge pixel in the image.

    Parameters:
    template (numpy.ndarray): The input image.

    Returns:
    dict: The reference table.
    """
    Ix, Iy = sobel_filter(template)
    G, theta = gradient_intensity(Ix, Iy)

    # Define the center point of the template
    xc, yc = template.shape[0] // 2, template.shape[1] // 2

    # Initialize the reference table
    reference_table = {}
    for x in range(template.shape[0]):
        for y in range(template.shape[1]):
            if G[x, y] > 0:
                # Calculate the vector from the center point to the edge pixel
                r = np.sqrt((xc - x)**2 + (yc - y)**2)
                alpha = np.arctan2(yc - y, xc - x)

                # Store this vector in the table, indexed by the orientation of the edge pixel
                orientation = theta[x, y]
                if orientation not in reference_table:
                    reference_table[orientation] = []
                reference_table[orientation].append((r, alpha))
    return reference_table


def calculate_accumulator(image, reference_table):
    """
    Calculate the accumulator array for the given image and reference table.

    Parameters:
    image (numpy.ndarray): The input image.
    reference_table (dict): The reference table.

    Returns:
    numpy.ndarray: The accumulator array.
    """
    Ix, Iy = sobel_filter(image)
    G, theta = gradient_intensity(Ix, Iy)

    # Initialize the accumulator array
    accumulator = np.zeros_like(image, dtype=np.float64)

    # Get the indices of the edge pixels
    edge_pixels = np.argwhere(G > 0)

    for x, y in edge_pixels:
        # Get the orientation of the edge pixel
        orientation = theta[x, y]

        # Look up the corresponding (r, alpha) pairs in the r-table
        if orientation in reference_table:
            for r, alpha in reference_table[orientation]:
                # Calculate the candidate center point
                xc = int(x + r * np.cos(alpha))
                yc = int(y + r * np.sin(alpha))

                # Increment the vote for that candidate center point
                if 0 <= xc < image.shape[0] and 0 <= yc < image.shape[1]:
                    accumulator[xc, yc] += 1

    return accumulator


def find_best_match(accumulator):
    """
    Find the location with the highest vote in the accumulator.

    Parameters:
    accumulator (numpy.ndarray): The accumulator array.

    Returns:
    tuple: The location with the highest vote.
    """
    best_matched_center = np.unravel_index(accumulator.argmax(), accumulator.shape)
    return best_matched_center


def find_all_matches(accumulator, threshold_ratio=0.8):
    """
    Find all locations with votes in the accumulator.

    Parameters:
    accumulator (numpy.ndarray): The accumulator array.
    threshold_ratio (float): The threshold ratio for finding all matches.

    Returns:
    numpy.ndarray: The locations with votes.
    """
    threshold = threshold_ratio * np.max(accumulator)
    matched_centers = np.argwhere(accumulator >= threshold)
    return matched_centers


def convolution_padding(image, kernel):
    """
    Manually convolve an image with a kernel.

    Parameters:
    image (numpy.ndarray): The input image.
    kernel (numpy.ndarray): The kernel.

    Returns:
    numpy.ndarray: The convolved image.
    """
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    output = np.zeros_like(image, dtype=np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Extract the region of interest
            region = padded_image[x:x + kernel_height, y:y + kernel_width]
            # Perform element-wise multiplication and sum the result
            output[x, y] = np.sum(region * kernel)
    return output


def sobel_filter(image):
    """
    Apply Sobel filter to the image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    tuple: The gradients in the x and y directions.
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    Ix = convolution_padding(image, Kx)
    Iy = convolution_padding(image, Ky)

    return Ix, Iy


def gradient_intensity(Ix, Iy):
    """
    Calculate the gradient intensity of the image.

    Parameters:
    Ix (numpy.ndarray): The gradient in the x direction.
    Iy (numpy.ndarray): The gradient in the y direction.

    Returns:
    tuple: The gradient intensity and the orientation.
    """
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)


def plotBestMatches(image, template, accumulator, matched_centers, best_match):
    """
    Plot the original image, template, accumulator, and matched locations.

    Parameters:
    image (numpy.ndarray): The original image.
    template (numpy.ndarray): The template.
    accumulator (numpy.ndarray): The accumulator array.
    matched_centers (numpy.ndarray): The matched centers.
    best_match (tuple): The best matched center.
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Original Image')

    ax[0, 1].imshow(template, cmap='gray')
    ax[0, 1].set_title('Template')

    if len(accumulator.shape) == 3:
        # Display the maximum vote across all angles for each pixel
        ax[1, 0].imshow(np.max(accumulator, axis=2), cmap='hot')
    else:
        ax[1, 0].imshow(accumulator, cmap='hot')
    ax[1, 0].set_title('Accumulator')

    ax[1, 1].imshow(image, cmap='gray')
    for center in matched_centers:
        ax[1, 1].scatter(center[1], center[0], s=100, c='green', marker='x')
    ax[1, 1].scatter(best_match[1], best_match[0], s=100, c='red', marker='x')
    ax[1, 1].set_title('Matched Locations')

    # Save the plot
    plt.savefig('output.png')

    plt.show()


# Create the parser
parser = argparse.ArgumentParser(description="Generalized Hough Transform")

# Add the arguments
parser.add_argument('mainImageName', type=str, help='The main image file name')
parser.add_argument('referenceImageName', type=str, help='The reference image file name')
parser.add_argument('--threshold_ratio', type=float, default=0.8, help='The threshold ratio for finding all matches')

# Parse the arguments
args = parser.parse_args()

mainImageName = args.mainImageName
referenceImageName = args.referenceImageName

referenceImage = cv2.imread(referenceImageName)
mainImage = cv2.imread(mainImageName)

# Converting the RGB image to a grayscale image.
referenceImage = cv2.cvtColor(referenceImage, cv2.COLOR_RGB2GRAY)
mainImage = cv2.cvtColor(mainImage, cv2.COLOR_RGB2GRAY)

# Perform edge detection on the images.
template = cv2.Canny(referenceImage, 100, 200)
image = cv2.Canny(mainImage, 100, 200)

reference_table = buildingReferenceTable(template)

# Calculate the accumulator array
accumulator = calculate_accumulator(image, reference_table)

# Find the best matched location
best_matched_center = find_best_match(accumulator)

# Find all matched locations
matched_centers = find_all_matches(accumulator, args.threshold_ratio)
print(len(matched_centers))

# Plot the results
plotBestMatches(image, template, accumulator, matched_centers, best_matched_center)
