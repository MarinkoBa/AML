from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os


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
        y_pred : array
            Contains predicted labels by the model
        y_truth : array
            Contains the ground truth labels
        filename : str
            Name of the created png-file
        show_plt : bool
            Marks if the plot of the confusion matrix should be shown during run time

    """

    print(confusion_matrix(y_truth, y_pred, labels=[0, 1, 2]))
    conf_matrix = confusion_matrix(y_truth, y_pred, labels=[0, 1, 2])

    df_cm = pd.DataFrame(conf_matrix, index=[i for i in ['NORMAL', 'COVID-19', 'PNEUMONIA']],
                         columns=[i for i in ['NORMAL', 'COVID-19', 'PNEUMONIA']])
    plt.figure(figsize=(10, 7))

    ax = sn.heatmap(df_cm, annot=True, cbar=False, linewidths=.5, cmap='mako')

    if show_plt is True:
        plt.show()

    # save plt as png in log-folder
    path = os.getcwd()
    parent = os.path.dirname(path)

    plt.savefig(parent + '/log/' + filename)


def get_evaluation_metrics(y_pred, y_truth, filename):
    """
        Parameters
        ----------
        y_pred : array
            Contains predicted labels by the model
        y_truth : array
            Contains the ground truth labels
        filename : str
            Name of the created file, which contains the evaluation metrics
    """

    print(classification_report(y_truth, y_pred, target_names=['NORMAL', 'COVID-19', 'PNEUMONIA']))
    report = classification_report(y_truth, y_pred, target_names=['NORMAL', 'COVID-19', 'PNEUMONIA'], output_dict=True)
    df = pd.DataFrame(report).transpose()

    # save result as csv file in log folder
    path = os.getcwd()
    parent = os.path.dirname(path)

    df.to_csv(parent + '/log/' + filename)
