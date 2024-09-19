import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.utils import resample
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Troy Krupinski, Jordan Ibarra 
#CSCE 5215
#Project 1

# R1: Data Pre-processing

def load_and_preprocess_data(file_path):
    # Load data
    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    data = pd.read_csv(file_path, names=columns, header=None)

    # Check for missing values
    print("Missing values in each column:\n", data.isnull().sum())

    # Handle missing values
    if data.isnull().sum().sum() == 0:
        print("\nMissing values not found in the dataset. Imputation not necessary.")
        data_imputed = data
    else:
        print("\nMissing values found in the dataset. Imputing with mean values.")
        print("Before imputation:\n", data.describe().loc[['min', 'max']])
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)




    # Normalize features to the range [0, 1]
    X = data_imputed.drop(['Id', 'Type'], axis=1)
    y = data_imputed['Type']
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("\nFeature ranges after normalization:\n", X_normalized.describe().loc[['min', 'max']])

    return X_normalized, y


# Apply RBF kernel transformation
def apply_rbf_kernel(X, gamma='scale'):
    if gamma == 'scale':
        gamma = 1 / (X.shape[1] * X.var().mean())
    rbf_sampler = RBFSampler(n_components=X.shape[1], gamma=gamma, random_state=42)
    X_rbf = rbf_sampler.fit_transform(X)
    return pd.DataFrame(X_rbf, columns=[f'RBF_{i}' for i in range(X_rbf.shape[1])])


# Visualize t-SNE before and after kernel transformation
def visualize_tsne(X, y, title):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='Set1')
    plt.title(title)
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.legend(loc='best')
    plt.show()


# R2: Model Generation and Tuning

def evaluate_model(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


# Print F1 scores and explanation
def print_f1_scores(model_name, y_true, y_pred):
    f1_scores = f1_score(y_true, y_pred, average=None)
    f1_scores_str = np.array2string(f1_scores, separator=',', precision=2, suppress_small=True)
    print(f"{model_name} F1 Scores: {f1_scores_str}")
    print(classification_report(y_true, y_pred, zero_division=0))


def requirement_r2(X, y, X_rbf):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    # Feature Engineering: Polynomial Features (degree=2)
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Dimensionality Reduction: PCA (retain 95% variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_poly)

    # Linear Perceptron with L2 Regularization
    perceptron_results = {}
    for max_iter in [1000, 2000, 5000]:
        model = Perceptron(max_iter=max_iter, class_weight='balanced', penalty='l2', random_state=42)
        perceptron_results[f'Perceptron (max_iter={max_iter})'] = {
            'Before RBF': evaluate_model(model, X_pca, y, cv),
            'After RBF': evaluate_model(model, X_rbf, y, cv)
        }
    
    # Logistic Regression with L1 and L2 Regularization
    lr_results = {}
    for penalty in ['l1', 'l2']:
        for max_iter in [1000, 2000, 5000]:
            model = LogisticRegression(max_iter=max_iter, penalty=penalty, solver='saga', class_weight='balanced', random_state=42)
            lr_results[f'Logistic Regression (penalty={penalty}, max_iter={max_iter})'] = {
                'Before RBF': evaluate_model(model, X_pca, y, cv),
                'After RBF': evaluate_model(model, X_rbf, y, cv)
            }

    # Multilayer Perceptron with tuning for max_iter and n_iter_no_change (patience)
    mlp_results = {}
    for max_iter in [1000, 2000, 5000]:
        for n_iter_no_change in [10, 20, 50]:
            model = MLPClassifier(max_iter=max_iter, n_iter_no_change=n_iter_no_change, learning_rate_init=0.001, solver='adam', random_state=42)
            mlp_results[f'MLP (max_iter={max_iter}, n_iter_no_change={n_iter_no_change})'] = {
                'Before RBF': evaluate_model(model, X_pca, y, cv),
                'After RBF': evaluate_model(model, X_rbf, y, cv)
            }

    # Select best linear model (Perceptron or Logistic Regression)
    best_perceptron_key = max(perceptron_results, key=lambda k: perceptron_results[k]['After RBF'][0])
    best_lr_key = max(lr_results, key=lambda k: lr_results[k]['After RBF'][0])

    best_perceptron_score = perceptron_results[best_perceptron_key]['After RBF'][0]
    best_lr_score = lr_results[best_lr_key]['After RBF'][0]

    if best_perceptron_score > best_lr_score:
        best_linear_model = Perceptron(max_iter=int(best_perceptron_key.split('=')[1][:-1]), class_weight='balanced', penalty='l2', random_state=42)
        print(f"Best Linear Model: {best_perceptron_key} (Perceptron)")
        model_type = "Perceptron"
    else:
        penalty_type = best_lr_key.split('penalty=')[1].split(',')[0]
        max_iter_value = int(best_lr_key.split('max_iter=')[1].split(')')[0])
        best_linear_model = LogisticRegression(max_iter=max_iter_value, penalty=penalty_type, solver='saga', class_weight='balanced', random_state=42)
        print(f"Best Linear Model: {best_lr_key} (Logistic Regression)")
        model_type = "Logistic Regression"

    # Train the best linear model on RBF-transformed data
    best_linear_model.fit(X_rbf, y)
    y_pred_linear = best_linear_model.predict(X_rbf)
    print_f1_scores(f"Best Linear Model ({model_type}, after RBF)", y, y_pred_linear)

    # Tune and compare MLP for analysis, but only for comparison (not for deployment)
    best_mlp_key = max(mlp_results, key=lambda k: mlp_results[k]['After RBF'][0])
    max_iter_value = int(best_mlp_key.split('max_iter=')[1].split(',')[0])
    n_iter_value = int(best_mlp_key.split('n_iter_no_change=')[1].split(')')[0])
    best_mlp_model = MLPClassifier(max_iter=max_iter_value, n_iter_no_change=n_iter_value, learning_rate_init=0.001, solver='adam', random_state=42)
    
    # Train MLP on RBF-transformed data for comparison
    best_mlp_model.fit(X_rbf, y)
    y_pred_mlp = best_mlp_model.predict(X_rbf)
    print_f1_scores("Best MLP Model (for comparison)", y, y_pred_mlp)

    # Return the best linear model (Perceptron or Logistic Regression)
    return best_linear_model


# R3: Scaling Up the Dataset

def requirement_r3(X_rbf, y, best_model):
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_rbf, y, test_size=0.3, random_state=42)

    # 2. Resample the training set (double the size)
    X_resampled, y_resampled = resample(X_train, y_train, n_samples=len(X_train)*2, random_state=42)

    # 3. Retrain the best linear classifier
    best_model.fit(X_resampled, y_resampled)

    # 4. Deploy on test set and display F measure values
    y_pred = best_model.predict(X_test)
    print_f1_scores("Best Linear Model (after resampling)", y_test, y_pred)

    return X_test, y_test, best_model


# R4: Identifying Least Significant Predictors

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


# Main Execution

if __name__ == "__main__":
    file_path = "glass.data"  

    # R1: Data Preprocessing. application of RBF Kernel & Visualization
    X, y = load_and_preprocess_data(file_path)
    visualize_tsne(X, y, "t-SNE Before RBF Kernel")
    X_rbf = apply_rbf_kernel(X)
    visualize_tsne(X_rbf, y, "t-SNE After RBF Kernel")

    # R2: Generate models and evaluate them
    best_linear_model = requirement_r2(X, y, X_rbf)

    # R3: Scaling up the dataset and evaluate the best linear model
    X_test, y_test, scaled_model = requirement_r3(X_rbf, y, best_linear_model)

    # R4: Identify least significant predictors and retrain
    requirement_r4(X, y, best_linear_model)
