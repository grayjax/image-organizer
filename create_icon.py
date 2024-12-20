from PIL import Image, ImageDraw

# Create a new image with a white background
size = (256, 256)
img = Image.new('RGB', size, 'white')
draw = ImageDraw.Draw(img)

# Draw a simple folder icon
draw.rectangle([50, 100, 206, 206], fill='#FFD700')  # folder body
draw.rectangle([50, 80, 120, 100], fill='#FFD700')   # folder tab

# Draw some "images" inside the folder
draw.rectangle([70, 120, 100, 150], fill='#4169E1')  # blue "image"
draw.rectangle([110, 130, 140, 160], fill='#32CD32') # green "image"
draw.rectangle([150, 125, 180, 155], fill='#DC143C') # red "image"

# Save as ICO
img.save('icon.ico') 