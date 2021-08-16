from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Metrics to evaluate the different trained multi-classification models:
# - Recall
# - Specificity
# - Precision
# - F1-Score
# - Accuracy

y_true = [1, 1, 2, 0, 1, 0, 2]
y_pred = [1, 2, 2, 0, 1, 2, 1]

print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

# visualization of confusion matrix
df_cm = pd.DataFrame(conf_matrix, index=[i for i in ['NORMAL', 'COVID-19', 'PNEUMONIA']],
                     columns=[i for i in ['NORMAL', 'COVID-19', 'PNEUMONIA']])
plt.figure(figsize=(10, 7))

ax = sn.heatmap(df_cm, annot=True, cbar=False, linewidths=.5, cmap='mako')
plt.show()

print(classification_report(y_true, y_pred, target_names=['NORMAL', 'COVID-19', 'PNEUMONIA']))
report = classification_report(y_true, y_pred, target_names=['NORMAL', 'COVID-19', 'PNEUMONIA'])
print('x')
