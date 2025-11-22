# import zipfile

# zip_path = "./synthetic_dataset_1.zip"
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall("synthetic_dataset_1")


train_path = "./dataset/synthetic_dataset_1/image_train"
val_path = "./dataset/synthetic_dataset_1/val"
output_folder = "./dataset/image_train_augmented"

# print folders length
import os
print("Train folder length:", len(os.listdir(train_path)))
print("Validation folder length:", len(os.listdir(val_path)))

import cv2
import albumentations as A

base_transform = A.Compose([
    A.RandomRain(
        slant_range=(-10, 10),
        drop_length=20,
        drop_width=1,
        drop_color=(180, 180, 180),
        blur_value=3,
        brightness_coefficient=0.9,
        rain_type="drizzle",
        p=0.3
    ),
    A.CLAHE(clip_limit=2, tile_grid_size=(4, 4), p=0.5),
    A.Emboss(alpha=(0.3, 0.5), strength=(0.3, 0.6), p=0.3),
    A.Equalize(mode="cv", by_channels=False, p=0.4),
    A.MotionBlur(blur_limit=(3, 7), p=0.3),
    A.RandomFog(alpha_coef=0.2, fog_coef_range=(0.05, 0.2), p=0.25),
    A.RandomGamma(gamma_limit=(80, 120), p=0.7),
    A.RandomSunFlare(
        flare_roi=(0, 0, 1, 0.5),
        src_radius=150,
        src_color=(255, 255, 255),
        num_flare_circles_range=(2, 4),
        p=0.2
    ),
    A.RingingOvershoot(blur_limit=(3, 7), cutoff=(0.5, 1.2), p=0.3),
    A.Sharpen(alpha=(0.05, 0.2), lightness=(0.7, 1.0), p=0.7),
    A.Spatter(
        mean=(0.4, 0.4),
        std=(0.2, 0.2),
        intensity=(0.1, 0.15),
        mode="rain",
        p=0.2
    ),
    A.Superpixels(
        p_replace=(0, 0.05),
        n_segments=(80, 120),
        max_size=100,
        interpolation=cv2.INTER_AREA,
        p=0.3
    )
])

def tilt(image, shear_range=(-10, 10)):
    transform = A.Affine(
        shear=shear_range,
        fit_output=True,
        p=1.0
    )
    return transform(image=image)["image"]

def perspective(image, scale=(0.02, 0.08)):
    transform = A.Perspective(
        scale=scale,
        keep_size=True,
        p=1.0
    )
    return transform(image=image)["image"]

def augment(image, use_tilt=False, use_perspective=False):
    img = base_transform(image=image)["image"]

    if use_tilt:
        img = tilt(img)

    if use_perspective:
        img = perspective(img)

    return img

from tqdm import tqdm

if __name__ == "__main__":
    # Example usage
    # img = cv2.imread("10A74930_10114.jpg")
    # out = augment(img, use_tilt=True, use_perspective=True)
    # cv2.imwrite("output.jpg", out)
    # print("Done! Saved output1.jpg")

    # Augment all train images and save them to a new folder
    os.makedirs(output_folder, exist_ok=True)

    import glob
    input_folder = train_path
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

    for img_path in tqdm(image_paths, desc="Augmenting images", unit="image"):
        img = cv2.imread(img_path)
        out = augment(img, use_tilt=True, use_perspective=True)
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, base_name), out)

    print("All images augmented and saved to", output_folder)
    print("Output folder length:", len(os.listdir(output_folder)))