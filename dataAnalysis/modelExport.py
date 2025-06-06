import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(123)

# Load your data (replace with your actual data loading)
df_extraction = pd.read_csv('dataAnalysis/data.csv')

X = df_extraction[['R', 'H', 'V', 'S', 'G', 'B', 'pH']]
y = df_extraction['Total Polyphenol Content']

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=None
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNet(alpha=0.2039565768208807, l1_ratio=0.4000859532936192, max_iter=4107))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'elasticnet_pipeline.pkl')