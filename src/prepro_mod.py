import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def prepro_data(fp):
    df = pd.read_csv(fp)

    if 'filename' in df.columns:  # removing unnecessary columns like 'filename'
        df = df.drop(['filename'], axis=1)
    
    X = df.drop(['label'], axis=1)     # dropping 'label' from X (feature) because it is what we need to predict (target-y)
    y = df['label']

    le = LabelEncoder()     # encoding the labels as some ml models cannot handle text labels (data)
    y_encoded = le.fit_transform(y)

    sclr = StandardScaler()
    X_scld = sclr.fit_transform(X)    # normalizing/standardizing the features for easy convergence

    X_train, X_test, y_train, y_test = train_test_split(X_scld, y_encoded, test_size=0.2, random_state=42)     # training and testing the data after spliting.

    return X_train, X_test, y_train, y_test, le
