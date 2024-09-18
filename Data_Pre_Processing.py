import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    # Load data
    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    data = pd.read_csv(file_path, names=columns, header=None)
    
    # Check for missing values
    print("Missing values:")
    print(data.isnull().sum())
    
    # Handle missing values if any
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Normalize features
    X = data_imputed.drop(['Id', 'Type'], axis=1)
    y = data_imputed['Type']
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print("\nFeature ranges after normalization:")
    print(X_normalized.describe().loc[['min', 'max']])
    
    return X_normalized, y

def apply_rbf_kernel(X, gamma='scale'):
    if gamma == 'scale':
        gamma = 1 / (X.shape[1] * X.var().mean())
    rbf_sampler = RBFSampler(n_components=X.shape[1], gamma=gamma, random_state=42)
    X_rbf = rbf_sampler.fit_transform(X)
    return pd.DataFrame(X_rbf, columns=[f'RBF_{i}' for i in range(X_rbf.shape[1])])

def visualize_tsne(X, y, title):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='deep')
    plt.title(title)
    print("Requirment R1: t-SNE visualization")
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()

def evaluate_model(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

def requirement_r2(X, y, X_rbf):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    # 1. Linear Perceptron
    perceptron_results = {}
    for max_iter in [1000, 2000, 5000, 10000]:
        model = Perceptron(max_iter=max_iter, random_state=42)
        perceptron_results[f'Perceptron (max_iter={max_iter})'] = {
            'Before RBF': evaluate_model(model, X, y, cv),
            'After RBF': evaluate_model(model, X_rbf, y, cv)
        }
    
    # 2. Logistic Regression
    lr_results = {}
    for max_iter in [1000, 2000, 5000, 10000]:
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        lr_results[f'Logistic Regression (max_iter={max_iter})'] = {
            'Before RBF': evaluate_model(model, X, y, cv),
            'After RBF': evaluate_model(model, X_rbf, y, cv)
        }
    
    # 3. Multilayer Perceptron
    mlp_results = {}
    for max_iter in [1000, 2000, 5000]:
        for n_iter_no_change in [50, 100, 200]:
            model = MLPClassifier(max_iter=max_iter, n_iter_no_change=n_iter_no_change, random_state=42)
            mlp_results[f'MLP (max_iter={max_iter}, n_iter_no_change={n_iter_no_change})'] = {
                'Before RBF': evaluate_model(model, X, y, cv),
                'After RBF': evaluate_model(model, X_rbf, y, cv)
            }
    
    # 4. Compare performance
    best_perceptron = max(perceptron_results, key=lambda k: perceptron_results[k]['After RBF'][0])
    best_lr = max(lr_results, key=lambda k: lr_results[k]['After RBF'][0])
    best_mlp = max(mlp_results, key=lambda k: mlp_results[k]['After RBF'][0])
    
    print("Best Perceptron:", best_perceptron)
    print("Best Logistic Regression:", best_lr)
    print("Best MLP:", best_mlp)
    
    # Compare F1 scores
    best_linear = best_perceptron if perceptron_results[best_perceptron]['After RBF'][0] > lr_results[best_lr]['After RBF'][0] else best_lr
    best_linear_model = Perceptron(max_iter=int(best_linear.split('=')[1][:-1])) if 'Perceptron' in best_linear else LogisticRegression(max_iter=int(best_linear.split('=')[1][:-1]))
    
    mlp_params = dict(param.split('=') for param in best_mlp.split('(')[1][:-1].split(', '))
    mlp_model = MLPClassifier(max_iter=int(mlp_params['max_iter']), n_iter_no_change=int(mlp_params['n_iter_no_change']), random_state=42)
    
    best_linear_model.fit(X_rbf, y)
    mlp_model.fit(X_rbf, y)
    
    y_pred_linear = best_linear_model.predict(X_rbf)
    y_pred_mlp = mlp_model.predict(X_rbf)
    
    print("\nBest Linear Model F1 Scores:")
    print(f1_score(y, y_pred_linear, average=None))
    print("\nMLP F1 Scores:")
    print(f1_score(y, y_pred_mlp, average=None))
    
    return best_linear_model, mlp_model

# Requirement R3
def requirement_r3(X_rbf, y, best_model):
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_rbf, y, test_size=0.3, random_state=42)
    
    # 2. Resample to double the size of the training set
    X_resampled, y_resampled = resample(X_train, y_train, n_samples=len(X_train)*2, random_state=42)
    
    # 3. Retrain the best linear classifier
    best_model.fit(X_resampled, y_resampled)
    
    # 4. Deploy on test set and display F measure values
    y_pred = best_model.predict(X_test)
    f1_scores = f1_score(y_test, y_pred, average=None)
    
    print("\nF1 Scores after scaling up the dataset:")
    print(f1_scores)
    
    return X_test, y_test, best_model

# Requirement R4
def requirement_r4(X, y, best_model):
    # 1. Identify two least significant predictors
    rfe = RFE(estimator=best_model, n_features_to_select=len(X.columns)-2, step=1)
    rfe = rfe.fit(X, y)
    
    feature_importance = pd.Series(rfe.ranking_, index=X.columns)
    least_significant = feature_importance.nlargest(2)
    
    print("\nTwo least significant predictors:")
    print(least_significant)
    
    
    # 3. Drop features and regenerate model
    X_reduced = X.drop(columns=least_significant.index)
    best_model.fit(X_reduced, y)
    y_pred = best_model.predict(X_reduced)
    f1_scores = f1_score(y, y_pred, average=None)
    
    print("\nF1 Scores after dropping least significant features:")
    print(f1_scores)

# Main execution
if __name__ == "__main__":
    file_path = "glass.data"  
    X, y = load_and_preprocess_data(file_path)
    X_rbf = apply_rbf_kernel(X)
    visualize_tsne(X, y, "t-SNE Visualization Before RBF")
    visualize_tsne(X_rbf, y, "t-SNE Visualization After RBF")
    print("Requirement R2:")
    best_linear_model, mlp_model = requirement_r2(X, y, X_rbf)
    
    print("\nRequirement R3:")
    X_test, y_test, scaled_model = requirement_r3(X_rbf, y, best_linear_model)
    
    print("\nRequirement R4:")
    requirement_r4(X, y, best_linear_model)