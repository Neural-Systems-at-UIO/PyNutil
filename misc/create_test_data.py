from PIL import Image, ImageDraw


def generate_test_image(width, height, file_path):
    # Create an empty white image
    image = Image.new(mode="RGB", size=(width, height), color=(255, 255, 255))

    # Save the image as a PNG file
    image.save(file_path)


# Example usage
width = 1000  # Specify the width of the image in pixels
height = 1000  # Specify the height of the image in pixels
file_path = "testimage.png"  # Specify the file path for saving the image

# Generate and save the black and white image
generate_test_image(width, height, file_path)


"""This is used to generate the test data"""


def generate_image_with_squares(
    width, height, square_diameter, square_locations, num_images
):
    # Create a white image with objects of specified size at specified x, y locations
    for i in range(1, num_images + 1):
        image = Image.new("RGB", (width, height), color=(255, 255, 255))

        # Create a draw object
        draw = ImageDraw.Draw(image)

        # Draw black squares
        for location in square_locations:
            x, y = location
            square = (x, y, x + (square_size - 1), y + (square_size - 1))
            # square defines the bounding box
            draw.rectangle(square, fill=(0, 0, 0))

        file_name = f"../test_data/PyTest/test_s00{i}.png"
        image.save(file_name, "PNG")


# Example usage
width = 1500  # Specify the width of the image in pixels
height = 1000  # Specify the height of the image in pixels
square_diameter = 10  # Specify the size of the black squares in pixels
square_locations = [
    (500, 500),
    (500, 600),
    (500, 700),
    (1000, 500),
    (1000, 600),
    (1000, 700),
]  # Specify the x, y locations of the black squares
num_images = 5

# Generate the white image with black squares
image = generate_image_with_squares(
    width, height, square_diameter, square_locations, num_images
)
