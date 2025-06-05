import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=123)

# Initialize metrics dataframe
metrics_df = pd.DataFrame(columns=['Model', 'RMSE', 'RRMSE', 'KGE', 'R2'])

# Utility functions
def remove_outliers(y_true, y_pred):
    """Remove outliers using 2 standard deviations threshold"""
    data = pd.DataFrame({'Observed': y_true, 'Predicted': y_pred})
    data['Difference'] = np.abs(data['Observed'] - data['Predicted'])
    threshold = data['Difference'].mean() + 2 * data['Difference'].std()
    data_clean = data[data['Difference'] <= threshold]
    return data_clean['Observed'].values, data_clean['Predicted'].values

def calculate_rrmse(y_true, y_pred):
    """Calculate Relative Root Mean Square Error"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / np.mean(y_true) * 100

def calculate_kge(y_true, y_pred):
    """Calculate Kling-Gupta Efficiency"""
    r = stats.pearsonr(y_true, y_pred)[0]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model and add to metrics dataframe"""
    global metrics_df
    
    # Remove outliers
    y_true_clean, y_pred_clean = remove_outliers(y_true, y_pred)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    rrmse = calculate_rrmse(y_true_clean, y_pred_clean)
    kge = calculate_kge(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Add to dataframe
    new_row = pd.DataFrame({
        'Model': [model_name],
        'RMSE': [rmse],
        'RRMSE': [rrmse],
        'KGE': [kge],
        'R2': [r2]
    })
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    
    return y_pred_clean

def create_scatter_plot(y_true, y_pred, title):
    """Create scatter plot for predictions vs observations"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='blue')
    plt.plot([0, 900], [0, 900], 'r-', linewidth=2)
    plt.xlim(0, 900)
    plt.ylim(0, 900)
    plt.xlabel('Observed TPC (mg GAE/L)')
    plt.ylabel('Predicted TPC (mg GAE/L)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt.gcf()

## 1. Gradient Boosting with Optuna -----
def optimize_gb(trial):
    """Optimize Gradient Boosting hyperparameters"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 123
    }
    
    model = GradientBoostingRegressor(**params)
    
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=cv, 
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()

print("Optimizing Gradient Boosting...")
study_gb = optuna.create_study(direction='minimize')
study_gb.optimize(optimize_gb, n_trials=100)

print(f"Best GB parameters: {study_gb.best_params}")
print(f"Best GB score: {study_gb.best_value}")

# Train final GB model
gb_model = GradientBoostingRegressor(**study_gb.best_params)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Evaluate and plot
gb_pred_clean = evaluate_model(y_test, gb_pred, "Gradient Boosting")
gb_plot = create_scatter_plot(y_test, gb_pred, "Gradient Boosting")

## 2. XGBoost with Optuna -----
def optimize_xgb(trial):
    """Optimize XGBoost hyperparameters"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 123
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=cv, 
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()

print("Optimizing XGBoost...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(optimize_xgb, n_trials=100)

print(f"Best XGBoost parameters: {study_xgb.best_params}")
print(f"Best XGBoost score: {study_xgb.best_value}")

# Train final XGBoost model
xgb_model = xgb.XGBRegressor(**study_xgb.best_params)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Evaluate and plot
xgb_pred_clean = evaluate_model(y_test, xgb_pred, "XGBoost")
xgb_plot = create_scatter_plot(y_test, xgb_pred, "XGBoost")

## 3. Elastic Net with Optuna -----
def optimize_elastic(trial):
    """Optimize Elastic Net hyperparameters"""
    params = {
        'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
        'max_iter': trial.suggest_int('max_iter', 1000, 5000),
        'random_state': 123
    }
    
    # Use StandardScaler for Elastic Net
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = ElasticNet(**params)
    
    # Cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, 
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()

print("Optimizing Elastic Net...")
study_elastic = optuna.create_study(direction='minimize')
study_elastic.optimize(optimize_elastic, n_trials=100)

print(f"Best Elastic Net parameters: {study_elastic.best_params}")
print(f"Best Elastic Net score: {study_elastic.best_value}")

# Train final Elastic Net model
scaler_elastic = StandardScaler()
X_train_scaled = scaler_elastic.fit_transform(X_train)
X_test_scaled = scaler_elastic.transform(X_test)

elastic_model = ElasticNet(**study_elastic.best_params)
elastic_model.fit(X_train_scaled, y_train)
elastic_pred = elastic_model.predict(X_test_scaled)

# Evaluate and plot
elastic_pred_clean = evaluate_model(y_test, elastic_pred, "Elastic Net")
elastic_plot = create_scatter_plot(y_test, elastic_pred, "Elastic Net")

## 4. SVM (Kernel Ridge) with Optuna -----
def optimize_svm(trial):
    """Optimize SVM hyperparameters"""
    params = {
        'C': trial.suggest_float('C', 0.01, 100, log=True),
        'gamma': trial.suggest_float('gamma', 0.001, 10, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.001, 1, log=True),
        'kernel': 'rbf'
    }
    
    # Use StandardScaler for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVR(**params)
    
    # Cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, 
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()

print("Optimizing SVM...")
study_svm = optuna.create_study(direction='minimize')
study_svm.optimize(optimize_svm, n_trials=100)

print(f"Best SVM parameters: {study_svm.best_params}")
print(f"Best SVM score: {study_svm.best_value}")

# Train final SVM model
scaler_svm = StandardScaler()
X_train_scaled = scaler_svm.fit_transform(X_train)
X_test_scaled = scaler_svm.transform(X_test)

svm_model = SVR(**study_svm.best_params)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

# Evaluate and plot
svm_pred_clean = evaluate_model(y_test, svm_pred, "SVM")
svm_plot = create_scatter_plot(y_test, svm_pred, "SVM")

## 5. Polynomial Regression with Optuna -----
def optimize_poly(trial):
    """Optimize Polynomial Regression hyperparameters"""
    degree = trial.suggest_int('degree', 1, 4)
    alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
    
    # Create polynomial pipeline
    poly_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('elastic', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=123))
    ])
    
    # Cross-validation
    scores = cross_val_score(poly_pipeline, X_train, y_train, cv=cv, 
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()

print("Optimizing Polynomial Regression...")
study_poly = optuna.create_study(direction='minimize')
study_poly.optimize(optimize_poly, n_trials=100)

print(f"Best Polynomial parameters: {study_poly.best_params}")
print(f"Best Polynomial score: {study_poly.best_value}")

# Train final Polynomial model
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=study_poly.best_params['degree'], include_bias=False)),
    ('elastic', ElasticNet(
        alpha=study_poly.best_params['alpha'],
        l1_ratio=study_poly.best_params['l1_ratio'],
        max_iter=2000,
        random_state=123
    ))
])

poly_pipeline.fit(X_train, y_train)
poly_pred = poly_pipeline.predict(X_test)

# Evaluate and plot
poly_pred_clean = evaluate_model(y_test, poly_pred, "Polynomial Regression")
poly_plot = create_scatter_plot(y_test, poly_pred, "Polynomial Regression")

## 6. Neural Network (MLP) with Optuna -----
def optimize_mlp(trial):
    """Optimize MLP hyperparameters"""
    # Hidden layer sizes
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_sizes = []
    for i in range(n_layers):
        size = trial.suggest_int(f'hidden_size_{i}', 10, 200)
        hidden_sizes.append(size)
    
    params = {
        'hidden_layer_sizes': tuple(hidden_sizes),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
        'max_iter': trial.suggest_int('max_iter', 200, 1000),
        'random_state': 123
    }
    
    # Use StandardScaler for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = MLPRegressor(**params)
    
    # Cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, 
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()

print("Optimizing Neural Network...")
study_mlp = optuna.create_study(direction='minimize')
study_mlp.optimize(optimize_mlp, n_trials=100)

print(f"Best MLP parameters: {study_mlp.best_params}")
print(f"Best MLP score: {study_mlp.best_value}")

# Train final MLP model
scaler_mlp = StandardScaler()
X_train_scaled = scaler_mlp.fit_transform(X_train)
X_test_scaled = scaler_mlp.transform(X_test)

# Reconstruct hidden layer sizes
n_layers = study_mlp.best_params['n_layers']
hidden_sizes = []
for i in range(n_layers):
    hidden_sizes.append(study_mlp.best_params[f'hidden_size_{i}'])

mlp_params = {
    'hidden_layer_sizes': tuple(hidden_sizes),
    'activation': study_mlp.best_params['activation'],
    'alpha': study_mlp.best_params['alpha'],
    'learning_rate_init': study_mlp.best_params['learning_rate_init'],
    'max_iter': study_mlp.best_params['max_iter'],
    'random_state': 123
}

mlp_model = MLPRegressor(**mlp_params)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)

# Evaluate and plot
mlp_pred_clean = evaluate_model(y_test, mlp_pred, "Neural Network")
mlp_plot = create_scatter_plot(y_test, mlp_pred, "Neural Network")

## Results Summary -----
print("\n" + "="*50)
print("OPTIMIZATION RESULTS SUMMARY")
print("="*50)
print(metrics_df.round(4))

# Save metrics to CSV
metrics_df.to_csv("dataAnalysis/model_metrics_optuna.csv", index=False)

# Create combined plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

models = ['Gradient Boosting', 'XGBoost', 'Elastic Net', 'SVM', 'Polynomial Regression', 'Neural Network']
predictions = [gb_pred, xgb_pred, elastic_pred, svm_pred, poly_pred, mlp_pred]

for i, (model, pred) in enumerate(zip(models, predictions)):
    ax = axes[i]
    ax.scatter(y_test, pred, alpha=0.6, s=30, color='blue')
    ax.plot([0, 900], [0, 900], 'r-', linewidth=2)
    ax.set_xlim(0, 900)
    ax.set_ylim(0, 900)
    ax.set_xlabel('Observed TPC (mg GAE/L)')
    ax.set_ylabel('Predicted TPC (mg GAE/L)')
    ax.set_title(f'{model}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dataAnalysis/model_comparisons_optuna.png', dpi=300, bbox_inches='tight')
plt.show()

# Print best parameters for each model
print("\n" + "="*50)
print("BEST HYPERPARAMETERS")
print("="*50)
print(f"Gradient Boosting: {study_gb.best_params}")
print(f"XGBoost: {study_xgb.best_params}")
print(f"Elastic Net: {study_elastic.best_params}")
print(f"SVM: {study_svm.best_params}")
print(f"Polynomial: {study_poly.best_params}")
print(f"Neural Network: {study_mlp.best_params}")