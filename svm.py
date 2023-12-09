import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


def svm():
    X = pd.read_csv('xys/all_x_per_speaker.csv')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = pd.read_csv('xys/y_per_speaker.csv').values.ravel()

    # classifier with 10-fold CV
    svm = SVC(kernel='linear', C=1)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(svm, X, y, cv=cv)
    accuracy_scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')

    print(f'Average Accuracy: {np.mean(accuracy_scores)}')
    
    # confusion matrix
    cm = confusion_matrix(y, y_pred_cv)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Lose', 'Predicted Win'], yticklabels=['True Lose', 'True Win'])
    plt.title(f'Confusion Matrix for SVM')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


if __name__ == "__main__":
    svm()
