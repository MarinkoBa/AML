import cv2
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import image
import numpy as np
import os


def augment_data(normal_imgs, covid_imgs, pneumonia_imgs, plot=False):
    """
    Increases the amount of covid and normal images through augmentation. Methods used are: rotation (10 degrees), horizontal flip and zoom (0.4 factor).
    336 Covid, 1266 Normal and 3418 Pneunomia images. -> create 336x4 = 1344 Covid, 1344 Normal, 1344 Pneumonia
    Normal and Covid images will be augmentated in order to get min. 1344 images -> Pneumonia no need for augmentation,
    already 3418 images available

    Parameters
    ----------
    normal_imgs:            list
                            A list including all the paths of normal images
    covid_imgs:             list
                            A list including all the paths of covid images
    pneumonia_imgs:         list
                            A list including all the paths of pneumonia images
    plot                    boolean
                            If true a exemplary aumentation will be showed
    Returns
    -------
    all_images:             list
                            A list including the final undersampled dataset (after augmentation).
    """

    normal_aug_imgs = []
    normal_orig_imgs = []

    covid_aug_imgs = []
    covid_orig_imgs = []

    for normal_img_path in normal_imgs:

        normal_img = cv2.imread(normal_img_path)

        normal_img_rotated_path, normal_img_rotated = rotate_10_degrees(normal_img_path, normal_img)
        normal_img_horizontal_flip_path, normal_img_horizontal_flip = flip_horizontal(normal_img_path, normal_img)
        normal_img_zoom_path, normal_img_zoom = zoom_04(normal_img_path, normal_img)

        normal_orig_imgs.append(normal_img_path)
        normal_aug_imgs.append(normal_img_rotated_path)
        normal_aug_imgs.append(normal_img_horizontal_flip_path)
        normal_aug_imgs.append(normal_img_zoom_path)

        if plot:
            plot_augmentations(normal_img, normal_img_rotated, normal_img_horizontal_flip, normal_img_zoom)

    for covid_img_path in covid_imgs:
        covid_img = cv2.imread(covid_img_path)

        covid_img_rotated_path, covid_img_rotated = rotate_10_degrees(covid_img_path, covid_img)
        covid_img_horizontal_flip_path, covid_img_horizontal_flip = flip_horizontal(covid_img_path, covid_img)
        covid_img_zoom_path, covid_img_zoom = zoom_04(covid_img_path, covid_img)

        covid_orig_imgs.append(covid_img_path)
        covid_aug_imgs.append(covid_img_rotated_path)
        covid_aug_imgs.append(covid_img_horizontal_flip_path)
        covid_aug_imgs.append(covid_img_zoom_path)

    normal_imgs_final = list(np.hstack([normal_orig_imgs, normal_aug_imgs]))
    covid_imgs_final = list(np.hstack([covid_orig_imgs, covid_aug_imgs]))

    all_images = balance_dataset_undersampling(normal_imgs_final, covid_imgs_final, pneumonia_imgs)

    return all_images


def zoom_04(normal_img_path, normal_img):
    """
    Generates from the normal_img a new zoomed version (with scale factor 0.4).
    If image already exists in the folder, it will be loaded

    Parameters
    ----------
    normal_img_path:        String
                            Path to the normal image without augmentation
    normal_img:             Image
                            Source image which will be augmented
    Returns
    -------
    normal_img_zoom_path:   String
                            The path of the new, augmented image.
    normal_img_zoom:        Image
                            The new, augmented image.
    """

    additional_path_str = '_zoom_04'
    normal_img_zoom_path = normal_img_path[:-4] + additional_path_str + normal_img_path[-4:]

    if os.path.isfile(normal_img_zoom_path) == False:
        normal_img_zoom = cv2.resize(normal_img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
        save_img(normal_img_zoom, normal_img_zoom_path)
    else:
        normal_img_zoom = image.imread(normal_img_zoom_path)

    return normal_img_zoom_path, normal_img_zoom


def flip_horizontal(normal_img_path, normal_img):
    """
    Generates from the normal_img a new horizontally flipped version.
    If image already exists in the folder, it will be loaded

    Parameters
    ----------
    normal_img_path:        String
                            Path to the normal image without augmentation
    normal_img:             Image
                            Source image which will be augmented
    Returns
    -------
    normal_img_horizontal_flip_path:   String
                            The path of the new, augmented image.
    normal_img_horizontal_flip:        Image
                            The new, augmented image.
    """

    additional_path_str = '_horizontal_flip'
    normal_img_horizontal_flip_path = normal_img_path[:-4] + additional_path_str + normal_img_path[-4:]

    if os.path.isfile(normal_img_horizontal_flip_path) == False:
        normal_img_horizontal_flip = cv2.flip(normal_img, 1)
        save_img(normal_img_horizontal_flip, normal_img_horizontal_flip_path)
    else:
        normal_img_horizontal_flip = image.imread(normal_img_horizontal_flip_path)

    return normal_img_horizontal_flip_path, normal_img_horizontal_flip


def rotate_10_degrees(normal_img_path, normal_img):
    """
    Generates from the normal_img a new rotated (10 degrees to the left) version.
    If image already exists in the folder, it will be loaded

    Parameters
    ----------
    normal_img_path:        String
                            Path to the normal image without augmentation
    normal_img:             Image
                            Source image which will be augmented
    Returns
    -------
    normal_img_rotated_path:   String
                            The path of the new, augmented image.
    normal_img_rotated:        Image
                            The new, augmented image.
    """

    degree_rotation = 10

    additional_path_str = '_rotated_' + str(degree_rotation)
    normal_img_rotated_path = normal_img_path[:-4] + additional_path_str + normal_img_path[-4:]

    if os.path.isfile(normal_img_rotated_path) == False:

        (h, w) = normal_img.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), degree_rotation, 1.0)
        normal_img_rotated = cv2.warpAffine(normal_img, M, (w, h))
        save_img(normal_img_rotated, normal_img_rotated_path)
    else:
        normal_img_rotated = image.imread(normal_img_rotated_path)

    return normal_img_rotated_path, normal_img_rotated


def save_img(normal_img_augmented, normal_img_augmented_path):
    im = Image.fromarray(normal_img_augmented)
    im.save(normal_img_augmented_path)


def balance_dataset_undersampling(normal_imgs, covid_imgs, pneumonia_imgs):
    """
    Balances dataset through undersampling (the category of the images: normal, covid or pneumonia)
    which includes the lowest amount of images will be used as the reference. So many images will be
    chosen also from the other two categories, such that each category includes same amount. Shuffling is
    applied always in the same manner (through seed(0)).

    Parameters
    ----------
    normal_imgs:            list
                            A list including all the paths of normal images
    covid_imgs:             list
                            A list including all the paths of covid images
    pneumonia_imgs:         list
                            A list including all the paths of pneumonia images

    Returns
    -------
    all_images:             list
                            A list including the final undersampled dataset.

    """

    np.random.seed(0)  # use always same seed

    amount_norm_cov_pne_arr = [len(normal_imgs), len(covid_imgs), len(pneumonia_imgs)]
    min_amount = np.min(amount_norm_cov_pne_arr)  # get min amount for balancing

    random_normal_imgs = np.random.choice(normal_imgs, min_amount)
    random_covid_imgs = np.random.choice(covid_imgs, min_amount)
    random_pneumonia_imgs = np.random.choice(pneumonia_imgs, min_amount)

    all_images = list(np.hstack([random_normal_imgs, random_covid_imgs, random_pneumonia_imgs]))

    return all_images


def plot_augmentations(normal_img, normal_img_rotated, normal_img_horizontal_flip, normal_img_zoom):

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(normal_img)
    ax[0, 0].set_title('Original')

    ax[0, 1].imshow(normal_img_rotated)
    ax[0, 1].set_title('Rotated')

    ax[1, 0].imshow(normal_img_horizontal_flip)
    ax[1, 0].set_title('Horizontal Flip')

    ax[1, 1].imshow(normal_img_zoom)
    ax[1, 1].set_title('Zoom 0.4')

    plt.show()
