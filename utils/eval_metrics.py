from sklearn.metrics import confusion_matrix, classification_report

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
print(classification_report(y_true, y_pred, target_names=['class1', 'class2', 'class3']))
report = classification_report(y_true, y_pred, target_names=['class1', 'class2', 'class3'])
print('x')

