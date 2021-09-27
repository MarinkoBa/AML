from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os

import numpy as np
from PIL import Image

import matplotlib as mpl

font = {'size'  : 20}

mpl.rc('font', **font)


# Metrics to evaluate the different trained multi-classification models:
# - Recall
# - Specificity
# - Precision
# - F1-Score
# - Accuracy

def get_confusion_matrix(y_pred, y_truth, filename, show_plt=False):
    """
    Parameters
    ----------
    y_pred:                 array
                            Contains predicted labels by the model
    y_truth:                array
                            Contains the ground truth labels
    filename:               str
                            Name of the created png-file
    show_plt:               bool
                            Marks if the plot of the confusion matrix should be shown during run time
    """

    print(confusion_matrix(y_truth, y_pred, labels=[0, 1, 2]))
    conf_matrix = confusion_matrix(y_truth, y_pred, labels=[0, 1, 2])

    df_cm = pd.DataFrame(conf_matrix, index=[i for i in ['NORMAL', 'COVID-19', 'PNEUMONIA']],
                         columns=[i for i in ['NORMAL', 'COVID-19', 'PNEUMONIA']])

    plt.figure(figsize=(8, 7))
    g = sn.heatmap(df_cm, annot=True, cbar=False, linewidths=.5, cmap='Blues', fmt="g",
                   cbar_kws={"orientation": "horizontal"}, annot_kws={"size": 32}, )
    g.set_yticklabels(labels=g.get_yticklabels(), va='center')

    if show_plt is True:
        plt.show()

    # save plt as png in log-folder
    path = os.getcwd()

    out_file_path = os.path.join(path, 'eval_metrics_nice_plots_20')

    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)

    plt.savefig(os.path.join(out_file_path, filename))

    img = Image.open(os.path.join(out_file_path, filename))

    w, h = img.size
    img.crop((50, 70, w - 70, h - 30)).save(os.path.join(out_file_path, filename) + '_cut.png')


def get_evaluation_metrics(y_pred, y_truth, filename):
    """
    Parameters
    ----------
    y_pred:                 array
                            Contains predicted labels by the model
    y_truth:                array
                            Contains the ground truth labels
    filename:               str
                            Name of the created file, which contains the evaluation metrics
    """

    print(classification_report(y_truth, y_pred, target_names=['NORMAL', 'COVID-19', 'PNEUMONIA']))
    report = classification_report(y_truth, y_pred, target_names=['NORMAL', 'COVID-19', 'PNEUMONIA'], output_dict=True)
    df = pd.DataFrame(report).transpose()

    # save result as csv file in log folder
    path = os.getcwd()

    out_file_path = os.path.join(path, 'eval_metrics_nice_plots_20')

    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)

    save_df_as_img(df, os.path.join(out_file_path, filename))

    df.to_csv(os.path.join(out_file_path, filename))


def mergecells(table, ix0, ix1):
    """
    Merges cells of the table
    Parameters
    ----------
    table:                  pyplot table
                            Table in which cells will be merged
    ix0:                    int
                            Index of the first of two to be merged cell
    ix1:                    int
                            Index of the second of two to be merged cell
    """

    ix0,ix1 = np.asarray(ix0), np.asarray(ix1)
    d = ix1 - ix0
    if not (0 in d and 1 in np.abs(d)):
        raise ValueError("ix0 and ix1 should be the indices of adjacent cells. ix0: %s, ix1: %s" % (ix0, ix1))

    if d[0]==-1:
        edges = ('BRL', 'TRL')
    elif d[0]==1:
        edges = ('TRL', 'BRL')
    elif d[1]==-1:
        edges = ('BTR', 'BTL')
    else:
        edges = ('BTL', 'BTR')

    # hide the merged edges
    for ix,e in zip((ix0, ix1), edges):
        table[ix[0], ix[1]].visible_edges = e

    txts = [table[ix[0], ix[1]].get_text() for ix in (ix0, ix1)]
    tpos = [np.array(t.get_position()) for t in txts]

    # center the text of the 0th cell between the two merged cells
    trans = (tpos[1] - tpos[0])/2
    if trans[0] > 0 and txts[0].get_ha() == 'right':
        # reduce the transform distance in order to center the text
        trans[0] /= 2
    elif trans[0] < 0 and txts[0].get_ha() == 'right':
        # increase the transform distance...
        trans[0] *= 2

    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))

    # hide the text in the 1st cell
    txts[1].set_visible(False)


def save_df_as_img(df, out_file_path):
    """
    Saves dataframe (evaluation metrics dataframe) as image without white spaces on the borders
    Parameters
    ----------
    df:                     dataframe
                            Contains evaluation metrics
    out_file_path:          str
                            Output file path of the image
    """

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    df2 = df.iloc[:-2, :-1]
    df2 = df2.round(2)
    df2['accuracy'] = [df2.iloc[3, 0], df2.iloc[3, 0], df2.iloc[3, 0], df2.iloc[3, 0]]
    df2 = df2.iloc[:-1]

    tab = ax.table(cellText=df2.values, colLabels=['Precision', 'Recall', 'F1-score', 'Accuracy'],
                   rowLabels=['Normal', 'Covid-19', 'Pneumonia'], loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)

    mergecells(tab, (2, 3), (1, 3))
    mergecells(tab, (2, 3), (3, 3))

    tab.scale(0.8, 1.4)

    for i in range(4):
        for j in range(4):
            tab.get_celld()[(i, j)].set_width(0.16)

    tab[2, 3].visible_edges = 'LR'

    fig.tight_layout()

    plt.savefig(os.path.join(out_file_path) + '.png')

    img = Image.open(os.path.join(out_file_path) + '.png')

    w, h = img.size
    img.crop((30, 180, w - 120, h - 180)).save(os.path.join(out_file_path) + '_cut.png')
