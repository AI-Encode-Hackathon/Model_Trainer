from PIL import Image
import numpy as np

def create_image_embedding(file):
    image = Image.open(file)

    image = image.resize((64, 64))
    image_array = np.array(image)
    emb = image_array.flatten()

    return emb
