from prepro_mod import prepro_data   # using the custom function (prepro_data) from preprocessing module (prepro_mod)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib     # to save models to file to use them again later (reusability)
from sklearn.model_selection import GridSearchCV
import os   # to work with file paths and folders

if not os.path.exists("models"):    # checking whether the folder exists or not
    os.makedirs("models")           # creates one if it does not exist

# loading the preprocessed dataset
X_train, X_test, y_train, y_test, le = prepro_data("D:\\University\\SEM-IV\\Applied ML\\AML Project\\Music-Genre-Classification\\Data\\features_30_sec.csv")

mod_configs = {       # model configurations and hyperparameters dictionary
    "KNN" : {
        "model" : KNeighborsClassifier(),         
        "params" : {"n_neighbors": [3, 5, 7]}     
    }, 
    "Logistic Regression" : {
        "model" : LogisticRegression(max_iter=1000),
        "params" : {"C": [0.1, 1.0, 10.0]}
    },
    "Decision Tree" : {
        "model" : DecisionTreeClassifier(),
        "params" : {"max_depth" : [None, 10, 20], "criterion" : ["gini", "entropy"]}
    },
    "Naive Bayes" : {
        "model" : GaussianNB(),
        "params" : {} 
    },
    "SVM" : {
        "model" : SVC(),
        "params" : {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}
    },
    "Random Forest" : {
        "model" : RandomForestClassifier(),
        "params" : {"n_estimators": [50, 100], "max_depth": [None, 20]}
    }
}

best_mods = {}   # creating an empty dictionary to store the best models

print("Now, starting with cross-validation and hyperparameter tuningon the trained models of each algorithms.")

for name, config in mod_configs.items():   # start training and hyperparameter tuning (goes through each model and its parameters)
    print (f" {name}")
    grd = GridSearchCV(config["model"], config["params"], cv=5, scoring= "accuracy", n_jobs=-1)   # performs cross-validation and finds the best combination of parameters based on accuracy
    grd.fit(X_train, y_train)   # trains the model based on these combinations

    print(f"Best Parameters: {grd.best_params_}")
    print(f"Best cross-validation accurracy: {grd.best_score_:.4f}\n")

    best_mods[name] = grd.best_estimator_       # stores the best model in the dictionary created earlier
    joblib.dump(grd.best_estimator_, f"models/{name.replace('', '_')}_mod.pkl")    # saves it as a pickle file

print("All the models are trained and saved succesfully!")

