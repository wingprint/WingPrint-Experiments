import glob
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from tqdm import tqdm

# seed for all pseudo-random operations
SEED = 57

# this transform is used to convert the PIL images to tensors when calculating the mean and standard deviation of the pixel values
_calc_transform = v2.Compose([
    v2.PILToTensor(),
    v2.ConvertImageDtype(torch.float)
])

def calc_mean_std_from_paths(input_dirs, limit=None):
    """
    Calculates the mean and standard deviation of the pixel values of all PNG images in input_dir and its subfolders.

    :param input_dir: Root directory. All PNG images in the root directory and any subdirectories will be included in the calculation.
    :param limit: Limits the number files that are used for the calculation. Speeds up the process a lot e.g. for debugging purposes.
    """

    if limit is None:
        print(f"calculate mean and std of {input_dirs}")
    else:
        print(f"calculate mean and std of {input_dirs} (limited to {limit} files)")

    channels_sum = torch.zeros(3) # we track three color channels for compatibility reasons (in our case one channel would be enough since we are dealing with greyscale images)
    channels_squared_sum = torch.zeros(3)
    num_pixels = 0

    images = []

    for input_dir in input_dirs:
        images.extend(glob.glob(os.path.join(input_dir, '**', '*.png'), recursive=True))

    if limit is not None:
        images = random.sample(images, k=limit)

    for file in tqdm(images):
        image = Image.open(file)

        batch = _calc_transform(image).unsqueeze(0)

        # sum the pixel values and squared pixel values across batch, height, and width
        channels_sum += torch.sum(batch, dim=[0, 2, 3])  # sum over batch, height, and width
        channels_squared_sum += torch.sum(batch ** 2, dim=[0, 2, 3])

        num_pixels += batch.size(2) * batch.size(3)  # H * W

    mean = channels_sum / num_pixels
    std = torch.sqrt(channels_squared_sum / num_pixels - mean ** 2)

    print(f"Calculated mean: {mean}, std: {std}")

    return mean, std

def get_device():
    """
    Returns a CUDA device or the CPU if no CUDA device is available.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"found CUDA device {torch.cuda.get_device_name(device)}")
        return device
    else:
        print(f"no CUDA device available, using CPU")
        return torch.device("cpu")
    
def seed_everything():
    """
    Seeds all randomization components and disables non-deterministic algorithms in PyTorch and CUDA.
    """

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def build_gallery_queries(device, model, gallery_dir, query_dir, gallery_transform, query_transform):
    """
    Builds a gallery and generates queries from the given data directories.

    :param device: (CUDA) device to use
    :param model: trained model to use for inference
    :param gallery_dir: directory that contains the gallery approaches
    :param query_dir: directory that containts the query approaches
    :param gallery_transform: input transform for the gallery images
    :param query_transform: input transform for the query images
    """

    model.eval()

    print("build gallery")

    gallery_embeddings = []
    gallery_names = []

    class_folders = glob.glob(os.path.join(gallery_dir, '*'))
    class_folders.sort(key=lambda class_folder: int(os.path.basename(class_folder)))

    with torch.inference_mode():
        for class_folder in tqdm(class_folders):
            class_name = int(os.path.basename(class_folder))

            folders = glob.glob(os.path.join(class_folder, '*'))

            class_embeddings = []

            for folder in folders:
                imgs = glob.glob(os.path.join(folder, '*.png'))

                for img in imgs:
                    pil_image = Image.open(img)

                    batch = gallery_transform(pil_image).unsqueeze(0)
                    batch = batch.to(device)

                    embedding = F.normalize(model.extractor(batch)).squeeze(0).cpu().detach().numpy()

                    class_embeddings.append(embedding)

            centroid = np.mean(class_embeddings, axis=0)

            gallery_embeddings.append(centroid)
            gallery_names.append(class_name)

    print("generate queries")

    query_embeddings = []
    query_names = []

    class_folders = glob.glob(os.path.join(query_dir, '*'))
    class_folders.sort(key=lambda class_folder: int(os.path.basename(class_folder)))

    with torch.inference_mode():
        for class_folder in tqdm(class_folders):
            class_name = int(os.path.basename(class_folder))

            folders = glob.glob(os.path.join(class_folder, '*'))

            for folder in folders:
                imgs = glob.glob(os.path.join(folder, '*.png'))

                approach_embeddings = []

                for img in imgs:
                    pil_image = Image.open(img)

                    batch = query_transform(pil_image).unsqueeze(0)
                    batch = batch.to(device)

                    embedding = F.normalize(model.extractor(batch)).squeeze(0).cpu().detach().numpy()

                    approach_embeddings.append(embedding)

                centroid = np.mean(approach_embeddings, axis=0)

                query_embeddings.append(centroid)
                query_names.append(class_name)

    return gallery_embeddings, gallery_names, query_embeddings, query_names