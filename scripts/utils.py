import numpy as np
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
from haar_filter import HaarFilter, Rectangle
import os
import pickle
import pandas as pd

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def image_loader(image_path):
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    if image.max() > 1:
        image = np.divide(image, 255.)
    return image


def normalize_image(image):
    if len(image.shape) == 2:
        image_max = np.max(image)
        image /= max(image_max, 1)
        return image
    elif len(image.shape) == 3:
        image_max = np.max(image, axis=(1, 2))
        ones = np.ones(len(image_max))
        image_max = np.where(image_max > 1, image_max, ones)
        return np.divide(image, np.expand_dims(image_max, (1, 2)))
    else:
        raise("Unsupported image shape")


def integral_image(image):
    if len(image.shape) == 2:
        cumsum_image = np.cumsum(image, axis=1)
        cumsum_image = np.cumsum(cumsum_image, axis=0)
        return cumsum_image
    elif len(image.shape) == 3:
        cumsum_image = np.cumsum(image, axis=2)
        cumsum_image = np.cumsum(cumsum_image, axis=1)
        return cumsum_image
    else:
        raise ("Unsupported image shape")

def resize_image(image, resolution=(24, 24)):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)
    return np.array(image.resize(size=resolution))


def crop_square(image, start_x, start_y, W, H):
    max_size = max(W, H)
    min_size = min(W, H)
    size = max_size
    if image.shape[0] <= start_y + size or image.shape[1] <= start_x + size:
        size = min_size
    return image[start_y: start_y + size, start_x: start_x + size]


def crop_and_resize(image, start_x, start_y, W,  H, output_resolution=(24, 24)):
    crop_im = crop_square(image, start_x, start_y, W, H)
    return resize_image(crop_im, resolution=output_resolution)


def random_crop(image, size, output_resolution=(24, 24)):
    assert image.shape[0] >= size and image.shape[1] >= size
    Y = np.random.randint(0, image.shape[0] + 1 - size)
    X = np.random.randint(0, image.shape[1] + 1 - size)
    return resize_image(image[Y: Y + size, X: X + size], output_resolution)


def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    assert pgmf.readline() == 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return np.array(raster)


def get_random_point(min_value, max_value):
    px1, px2 = np.random.randint(min_value, max_value + 1, 2)
    return min(px1, px2), max(px1, px2)


def generate_random_filter(H, W, actual_width=20, actual_height=20, n_vertical_split=0, n_horizontal_split=1):
    assert actual_height <= H and actual_width <= W
    start_y, end_y = get_random_point(0, H - actual_height)
    start_x, end_x = get_random_point(0, W - actual_width)
    rectangle_h = actual_height // (n_vertical_split + 1)
    rectangle_w = actual_width // (n_horizontal_split + 1)

    color = 1
    rectangles = []
    for i in range(0, n_vertical_split + 1):
        color += 1
        color %= 2
        for j in range(0, n_horizontal_split + 1):
            x, y = start_x + rectangle_w * j, start_y + rectangle_h * i
            rectangles.append(Rectangle(top_left=[x, y], H=rectangle_h, W=rectangle_w, color=color))
            if n_horizontal_split:
                color += 1
                color %= 2

    return HaarFilter(H=H, W=W, rectangles=rectangles)


def generate_filters(H, W, n, output_path="filters", min_height=4, max_height=10, min_width=4, max_width=10):
    assert min_height >= 3 and min_width >= 3
    assert max_height >= min_height and max_width >= min_width
    os.makedirs(output_path,  exist_ok=True)
    filters = []
    i = 0
    while i < n:
        actual_height = np.random.randint(min_height, max_height + 1)
        actual_width = np.random.randint(min_width, max_width + 1)
        n_horizontal_split = np.random.randint(0, 3)
        if not n_horizontal_split:
            n_vertical_split = np.random.randint(1, 3)
        elif n_horizontal_split == 1:
            n_vertical_split = np.random.randint(0, 2)
        elif n_horizontal_split == 2:
            n_vertical_split = 0
        else:
            raise Exception(f"{n_horizontal_split} is not supported.")
        f = generate_random_filter(H=H, W=W, actual_width=actual_width, actual_height=actual_height,
                                   n_vertical_split=n_vertical_split, n_horizontal_split=n_horizontal_split)
        if f not in filters:
            filters.append(f)
            i += 1
        else:
            print(f"number of filters: {len(filters)}")

    for idx, f in enumerate(filters):
        f.save(os.path.join(output_path, f"filter_{idx}.jpg"))

    with open("../pickle/filters.pkl", "wb") as f:
        pickle.dump(filters, f)


def load_custom_database(n=-1):
    df = pd.read_csv("../face.csv")
    non_face_path = os.path.join("../data", "no_face")
    debug_images = []
    all_labels = []
    faces = []
    for image_name in df.name.values[:n]:
        X = df[df.name == image_name].x.values[0]
        Y = df[df.name == image_name].y.values[0]
        H = df[df.name == image_name].h.values[0]
        W = df[df.name == image_name].w.values[0]
        image = image_loader(os.path.join("../data", "face", image_name))
        face = crop_and_resize(image, X, Y, W, H, output_resolution=(24, 24))
        debug_images.append(face)
        face = normalize_image(face)
        face = integral_image(face)
        face = np.expand_dims(face, 0)
        faces.append(face)
        all_labels.append(1)
    all_images = np.concatenate(faces, axis=0)
    print(f"Number of images {len(all_images)}")
    non_face = [path for path in os.listdir(non_face_path)]
    M = len(non_face)
    n = n if n >= 0 else M
    N = max(len(all_images), M)
    N *= 2
    n = int(N // M) if N > M else 1
    non_faces = []
    for image_name in non_face[:N]:
        image = image_loader(os.path.join(non_face_path, image_name))
        for i in range(n):
            crop_image = random_crop(image, size=600, output_resolution=(24, 24))
            crop_image = normalize_image(crop_image)
            crop_image = integral_image(crop_image)
            crop_image = np.expand_dims(crop_image, 0)
            non_faces.append(crop_image)
            all_labels.append(-1)

    non_faces = np.concatenate(non_faces, axis=0)
    all_images = np.concatenate([all_images, non_faces], axis=0)
    all_labels = np.array(all_labels)
    print(f"Images shape {all_images.shape}")
    print(f"Labels shape {all_labels.shape}")
    data = {"images": all_images, "labels": all_labels}
    with open("../data/train_data.pkl", "wb") as f:
        pickle.dump(data, f)

    return all_images, all_labels


def load_train_database(cache=False):
    if cache:
        data = load_pickle("../data/train_data.pkl")
        return data["images"], data["labels"]

    all_images, all_labels = load_custom_database(n=-1)
    pos_images = [np.expand_dims(integral_image(normalize_image(image_loader(f"face-detection-data/pos/{x}"))), 0)
                  for x in os.listdir("../face-detection-data/pos")]
    neg_images = [np.expand_dims(integral_image(normalize_image(image_loader(f"face-detection-data/neg/{x}"))), 0)
                  for x in os.listdir("../face-detection-data/neg") if x.endswith("24x24.pgm")]
    # pos_labels = np.ones(len(pos_images))
    neg_labels = np.ones(len(neg_images)) * -1
    # all_images = np.concatenate([all_images, np.concatenate(pos_images, axis=0), np.concatenate(neg_images, axis=0)], axis=0)
    # all_labels = np.concatenate([all_labels, pos_labels, neg_labels])
    all_images = np.concatenate([all_images, np.concatenate(neg_images, axis=0)], axis=0)
    all_labels = np.concatenate([all_labels, neg_labels])
    print(f"Images shape {all_images.shape}")
    print(f"Labels shape {all_labels.shape}")
    data = {"images": all_images, "labels": all_labels}
    with open("../data/train_data.pkl", "wb") as f:
        pickle.dump(data, f)

    return all_images, all_labels


def load_test_database():
    pos_images = [np.expand_dims(integral_image(normalize_image(image_loader(f"face-detection-data/pos/{x}"))), 0)
                  for x in os.listdir("../face-detection-data/pos")]
    neg_images = [np.expand_dims(integral_image(normalize_image(image_loader(f"face-detection-data/neg/{x}"))), 0)
                  for x in os.listdir("../face-detection-data/neg") if x.endswith("24x24.pgm")]
    pos_labels = np.ones(len(pos_images))
    neg_labels = np.ones(len(neg_images)) * -1
    all_images = np.concatenate([np.concatenate(pos_images, axis=0), np.concatenate(neg_images, axis=0)], axis=0)
    all_labels = np.concatenate([pos_labels, neg_labels])
    print(f"Images shape {all_images.shape}")
    print(f"Labels shape {all_labels.shape}")
    data = {"images": all_images, "labels": all_labels}
    with open("../data/test_data.pkl", "wb") as f:
        pickle.dump(data, f)

    return all_images, all_labels
