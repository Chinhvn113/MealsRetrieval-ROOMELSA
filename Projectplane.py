import imageio
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import os
from datetime import datetime
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg" # change it with your ffmpeg dir
from tqdm import tqdm

def map_to_sphere(x, y, z, yaw_radian, pitch_radian):


    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    # Apply rotation transformations here
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                           np.cos(theta) * np.sin(pitch_radian),
                           np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')

    return output_image

# Generate 20 images with different perspectives
# Create a unique output folder for each run
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
panorama_path = "Image-20250404T180246Z-001/Image/1_colors.png"
output_folder = f"output_{panorama_path.split('/')[2]}_{timestamp.split('-')[1]}"
os.makedirs(output_folder, exist_ok=True)

print(f"Saving images to: {output_folder}")

# Generate 20 images with different perspectives
for i, deg in enumerate(np.linspace(0, 360, 20)):  # 20 different angles
    output_image = panorama_to_plane(
        panorama_path,
        90, (1200, 1200), deg, 90
    )
    output_image.save(f"{output_folder}/perspective_{i:02d}.png")

print(f"Saved 20 images in '{output_folder}'")