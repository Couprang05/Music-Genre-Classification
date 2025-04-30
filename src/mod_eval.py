from prepro_mod import prepro_data   # using the custom function (prepro_data) from preprocessing module (prepro_mod)
import joblib   # to save models to file to use them again later (reusability)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# loading the preprocessed dataset.
X_train, X_test, y_train, y_test, le = prepro_data("D:\\University\\SEM-IV\\Applied ML\\AML Project\\Music-Genre-Classification\\Data\\features_30_sec.csv")

mod_names = [       # list of all models trained earlier
    "KNN",
    "Logistic Regression",
    "Decision Tree",
    "Naive Bayes",
    "SVM",
    "Random Forest"
]

# creating empty dictionaries to store the scores of the evaluation metrics for these models to compare them later
acc_scores = {}         # accuracy 
prec_scores = {}        # precision 
rec_scores = {}         # recall
f1_scores = {}          # F1-score

for name in mod_names:  # looping through each model to load, predict, and evaluate
    print(f"\n Evaluating {name}")
    mod = joblib.load(f"models/{name.replace('', '_')}_mod.pkl")     # loads the saved models from the folder (models)
    y_pred = mod.predict(X_test)        # makes predictions on the test data

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')    # calculating the average score across all the genres (classes) equally
    rec = recall_score(y_test, y_pred, average='macro')       
    f1 = f1_score(y_test, y_pred, average='macro')
    
    acc_scores[name] = acc        # storing all the scores caluculated above
    prec_scores[name] = prec
    rec_scores[name] = rec
    f1_scores[name] = f1

    print(f"Accuracy: {acc:.4f}")      
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))    # prints the detailed report of the metrics

    cm = confusion_matrix(y_test, y_pred)      # creates a heatmap plot for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=list(acc_scores.keys()), y=list(acc_scores.values()))     # plotting a bar chart to show the comparison of accuracy of models  
plt.title("Model Accuracy Comparison (Test Set)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

metrics_df = pd.DataFrame({       # creating a dataframe to store the evaluation metrics
    "Precision": prec_scores,
    "Recall": rec_scores,
    "F1-Score": f1_scores
})

metrics_df.plot(kind="bar", figsize=(12, 6))    # plotting a multi-metric bar chart to show the comparison between precision, recall and f1 score of all the models
plt.title("Precision, Recall, and F1-Score Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()