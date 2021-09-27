from matplotlib import pyplot as plt
import matplotlib
import cv2

font = {'size'  : 18}

matplotlib.rc('font', **font)


def plot_normal_covid_pneumonia_examples(normal_img_path, covid_img_path, pneumonia_img_path):
    """
    Plots 3 images: 1x normal, 1x covid and 1x pneumonia from the corresponding paths
    Parameters
    ----------
    normal_img_path:        str
                            A path to the normal image
    covid_img_path:         str
                            A path to the COVID-19 image
    pneumonia_img_path:     str
                            A path to the Pneumonia image

    """

    normal_img = cv2.imread(normal_img_path)
    covid_img = cv2.imread(covid_img_path)
    pneumonia_img = cv2.imread(pneumonia_img_path)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(normal_img)
    ax[0].set_title('Normal')

    ax[1].imshow(covid_img)
    ax[1].set_title('Covid19')

    ax[2].imshow(pneumonia_img)
    ax[2].set_title('Pneumonia')

    plt.show()


def plot_augmentations(normal_img, normal_img_rotated, normal_img_horizontal_flip, normal_img_zoom):
    """
    Plots 4 images regarding the augmentation: 1x Original, 1x Rotated and 1x Horizontal Flipped
    and 1x Zoomed by factor 0.4
    Parameters
    ----------
    normal_img:                 array-like or PIL image
                                A normal image to be plotted
    normal_img_rotated:         array-like or PIL image
                                A rotated image to be plotted
    normal_img_horizontal_flip: array-like or PIL image
                                A horizontal flipped image to be plotted
    normal_img_zoom:            array-like or PIL image
                                A zoomed by 0.4 image to be plotted
    """

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
