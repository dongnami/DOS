from PIL import Image


def concatenate_images_with_spacing(images, spacing=20):
    """
    Concatenates a list of PIL images horizontally with a specified spacing between them.

    Args:
        images (list): A list of PIL.Image objects.
        spacing (int): The spacing (in pixels) between each image.

    Returns:
        PIL.Image: A new image with the input images concatenated horizontally.
    """
    # Calculate the total width and maximum height of the final image
    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
    max_height = max(img.height for img in images)
    
    # Create a new blank image with the calculated dimensions
    result_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))  # White background
    
    # Paste each image into the result image with spacing
    x_offset = 0
    for img in images:
        result_image.paste(img, (x_offset, 0))
        x_offset += img.width + spacing
    
    return result_image
