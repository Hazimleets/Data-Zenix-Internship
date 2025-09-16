# src/evaluate.py

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import get_data_generators

# Paths
train_dir = "../data/seg_train/seg_train"
test_dir = "../data/seg_test/seg_test"
model_path = "../saved_models/intel_model.h5"

# Load model
model = load_model(model_path)

# Load test data
_, test_gen = get_data_generators(train_dir, test_dir)

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc*100:.2f}%")

# Predictions
y_pred = model.predict(test_gen)
y_pred_classes = y_pred.argmax(axis=1)
y_true = test_gen.classes

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=list(test_gen.class_indices.keys())))
