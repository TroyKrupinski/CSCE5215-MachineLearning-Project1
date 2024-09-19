import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    # Load data
    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    data = pd.read_csv(file_path, names=columns, header=None)
    
    # Check for missing values
    print("Missing values in each column:\n", data.isnull().sum())
    
    # Handle missing values only if any are present
    if data.isnull().sum().sum() == 0:
        print("No missing values found in the dataset!")
        data_imputed = data
    else:
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
    # Normalize features to the range [0, 1]
    X = data_imputed.drop(['Id', 'Type'], axis=1)
    y = data_imputed['Type']
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print("\nFeature ranges after normalization:\n", X_normalized.describe().loc[['min', 'max']])
    
    return X_normalized, y

def apply_rbf_kernel(X, gamma='scale'):
    # Applying RBF kernel to reduce overlap in feature space
    if gamma == 'scale':
        gamma = 1 / (X.shape[1] * X.var().mean())
    rbf_sampler = RBFSampler(n_components=X.shape[1], gamma=gamma, random_state=42)
    X_rbf = rbf_sampler.fit_transform(X)
    return pd.DataFrame(X_rbf, columns=[f'RBF_{i}' for i in range(X_rbf.shape[1])])

def visualize_tsne(X, y, title):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='Set1', style=y, legend='full')
    plt.title(title)
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.legend(loc='best')
    plt.show()

def evaluate_model(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

def print_f1_scores(model_name, y_true, y_pred):
    f1_scores = f1_score(y_true, y_pred, average=None)
    f1_scores_str = np.array2string(f1_scores, separator=',', precision=2, suppress_small=True)
    print(f"{model_name} F1 Scores: {f1_scores_str}")
    print(classification_report(y_true, y_pred))

def requirement_r2(X, y, X_rbf):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    # Linear Perceptron
    perceptron_results = {}
    for max_iter in [1000, 2000, 5000]:
        model = Perceptron(max_iter=max_iter, class_weight='balanced', random_state=42)
        perceptron_results[f'Perceptron (max_iter={max_iter})'] = {
            'Before RBF': evaluate_model(model, X, y, cv),
            'After RBF': evaluate_model(model, X_rbf, y, cv)
        }
    
    # Logistic Regression
    lr_results = {}
    for max_iter in [1000, 2000, 5000]:
        model = LogisticRegression(max_iter=max_iter, class_weight='balanced', random_state=42)
        lr_results[f'Logistic Regression (max_iter={max_iter})'] = {
            'Before RBF': evaluate_model(model, X, y, cv),
            'After RBF': evaluate_model(model, X_rbf, y, cv)
        }
    
    # Multilayer Perceptron
    mlp_results = {}
    for max_iter in [2000, 3000]:
        model = MLPClassifier(max_iter=max_iter, learning_rate_init=0.001, solver='adam', random_state=42)
        mlp_results[f'MLP (max_iter={max_iter})'] = {
            'Before RBF': evaluate_model(model, X, y, cv),
            'After RBF': evaluate_model(model, X_rbf, y, cv)
        }
    
    # Compare performance
    best_perceptron = max(perceptron_results, key=lambda k: perceptron_results[k]['After RBF'][0])
    best_lr = max(lr_results, key=lambda k: lr_results[k]['After RBF'][0])
    best_mlp = max(mlp_results, key=lambda k: mlp_results[k]['After RBF'][0])
    
    print(f"Best Perceptron: {best_perceptron}")
    print(f"Best Logistic Regression: {best_lr}")
    print(f"Best MLP: {best_mlp}")
    
    # Compare F1 scores for best models
    best_linear_model = Perceptron(max_iter=1000, class_weight='balanced') if 'Perceptron' in best_perceptron else LogisticRegression(max_iter=1000, class_weight='balanced')
    
    # Fitting the best models on RBF data
    best_linear_model.fit(X_rbf, y)
    y_pred_linear = best_linear_model.predict(X_rbf)
    print_f1_scores("Best Linear Model", y, y_pred_linear)
    
    best_mlp_model = MLPClassifier(max_iter=2000, learning_rate_init=0.001, solver='adam', random_state=42)
    best_mlp_model.fit(X_rbf, y)
    y_pred_mlp = best_mlp_model.predict(X_rbf)
    print_f1_scores("Best MLP Model", y, y_pred_mlp)
    
    return best_linear_model

# Requirement R3
def requirement_r3(X_rbf, y, best_model):
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_rbf, y, test_size=0.3, random_state=42)
    
    # 2. Resample the training set (SMOTE for class balance)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # 3. Retrain the best linear classifier
    best_model.fit(X_train_smote, y_train_smote)
    
    # 4. Deploy on test set and display F measure values
    y_pred = best_model.predict(X_test)
    print_f1_scores("Best Linear Model (after resampling)", y_test, y_pred)
    
    return X_test, y_test, best_model

# Requirement R4
def requirement_r4(X, y, best_model):
    # Feature Importance using Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    
    least_significant = feature_importance.nsmallest(2)
    print("\nLeast significant predictors:\n", least_significant)
    
    # Drop least significant predictors
    X_reduced = X.drop(columns=least_significant.index)
    best_model.fit(X_reduced, y)
    y_pred = best_model.predict(X_reduced)
    
    print_f1_scores("Best Model (after dropping least significant features)", y, y_pred)

# Main execution
if __name__ == "__main__":
    file_path = "glass.data"  # Update this with your actual dataset path
    X, y = load_and_preprocess_data(file_path)
    X_rbf = apply_rbf_kernel(X)
    
    # t-SNE Visualization before and after RBF kernel
    visualize_tsne(X, y, "t-SNE Visualization Before RBF")
    visualize_tsne(X_rbf, y, "t-SNE Visualization After RBF")
    
    print("\nRequirement R2:")
    best_linear_model = requirement_r2(X, y, X_rbf)
    
    print("\nRequirement R3:")
    X_test, y_test, scaled_model = requirement_r3(X_rbf, y, best_linear_model)
    
    print("\nRequirement R4:")
    requirement_r4(X, y, best_linear_model)
