# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# Load the dataset

crops = pd.read_csv("soil_measures.csv")



# Separate target
X = crops.drop('crop', axis=1)
y = crops["crop"]

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

features = ['N', 'P', 'K', 'ph']
scores = {}

for feature in features:
    # Train on only one feature at a time
    X_train_f = X_train[[feature]]
    X_test_f = X_test[[feature]]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_test_scaled = scaler.transform(X_test_f)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)

    scores[feature] = accuracy_score(y_test, y_pred)

print(scores)
best_predictive_feature = {}
best_predictive_feature['K']=scores['K']