from PIL import Image, ImageDraw

def generate_test_image(width, height, file_path):
    # Create an empty white image
    image = Image.new(mode = "RGB", size = (width, height), color=(255,255,255))

    # Save the image as a PNG file
    image.save(file_path)

# Example usage
width = 1000  # Specify the width of the image in pixels
height = 1000  # Specify the height of the image in pixels
file_path = 'testimage.png'  # Specify the file path for saving the image

# Generate and save the black and white image
generate_test_image(width, height, file_path)

def generate_white_image_with_squares(width, height, square_size, square_locations):
    # Create a white image with objects of specified size and x, y location
    image = Image.new('RGB', (width, height), color=(255, 255, 255))

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Draw black squares
    for location in square_locations:
        x, y = location
        square = (x, y, x + square_size, y + square_size)
        draw.rectangle(square, fill=(0, 0, 0))

    return image

# Example usage
width = 1000  # Specify the width of the image in pixels
height = 1000 # Specify the height of the image in pixels
square_size = 50  # Specify the size of the black squares in pixels
square_locations = [(500, 500), (150, 150), (250, 250), (750, 750)]  # Specify the x, y locations of the black squares

# Generate the white image with black squares
image = generate_white_image_with_squares(width, height, square_size, square_locations)

# Save the image
image.save('white_image_with_squares.png')