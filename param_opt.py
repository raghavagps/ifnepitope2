import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]  # Features (all columns except the last)
    y = data.iloc[:, -1]   # Labels (last column)
    return X, y

# Define the classifiers and their parameter grids
def get_classifiers_and_param_grids():
    classifiers = {
        'DT': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 50, 100],
            'min_samples_split': [10, 20, 30, 40, 50],
            'max_features': ['auto', 'sqrt', 'log2']
        }),
        'RF': (RandomForestClassifier(random_state=42, class_weight='balanced'), {
            'n_estimators': [10, 50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2']
        }),
        'LR': (LogisticRegression(random_state=42, class_weight='balanced'), {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['liblinear']
        }),
        'XGB': (XGBClassifier(random_state=42), {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
        }),
        'GNB': (GaussianNB(), {
            'var_smoothing': np.logspace(0, -9, num=100)
        }),
        'ET': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {
            'n_estimators': [10, 50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2']
        }),
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': np.arange(50, 100),
            'metric': ['minkowski', 'euclidean', 'manhattan'],
            'weights': ['uniform', 'distance']
        }),
        'SVC': (SVC(random_state=42, class_weight='balanced', probability=True), {
            'kernel': ['rbf', 'linear'],
            'gamma': [0.0005, 0.001, 0.05, 0.1],
            'C': [0.001, 0.01, 0.1, 1, 10]
        })
    }
    return classifiers

# Function to perform grid search and calculate AUROC
def perform_grid_search(X, y, classifier_name, classifier, param_grid):
    best_params_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Dynamically adjust all KNN parameters based on the number of samples in each fold
    if classifier_name == 'KNN':
        # Get the smallest training fold size from cross-validation
        smallest_fold_size = min(len(train_idx) for train_idx, _ in cv.split(X, y))

        # Adjust n_neighbors to be <= smallest fold size
        max_n_neighbors = min(smallest_fold_size, param_grid['n_neighbors'].max())
        param_grid['n_neighbors'] = np.arange(1, max_n_neighbors + 1)

        # Print the adjusted n_neighbors range for debugging
        print(f"Adjusted n_neighbors range for KNN: {param_grid['n_neighbors']}")

    try:
        print(f"Running Grid Search for {classifier_name}...")
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        y_proba = grid_search.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y, y_proba)
        
        best_params_results.append({
            'Classifier': classifier_name,
            'Best Parameters': grid_search.best_params_,
            'AUROC': auroc
        })
        
        print(f"Best parameters for {classifier_name}: {grid_search.best_params_} with AUROC: {auroc}")
    except Exception as e:
        print(f"Error with {classifier_name}: {e}")
    return best_params_results

# Save the best parameters and AUROC to a txt file
def save_best_params(best_params_results, output_file):
    with open(output_file, 'w') as f:
        f.write("Classifier\tBest Parameters\tAUROC\n")
        for result in best_params_results:
            f.write(f"{result['Classifier']}\t{result['Best Parameters']}\t{result['AUROC']:.4f}\n")

if __name__ == "__main__":
    # Setup argparse to get input arguments
    parser = argparse.ArgumentParser(description='Hypertune classifiers with GridSearchCV.')
    parser.add_argument('--file', type=str, required=True, help='Path to the input CSV file containing features and labels.')
    parser.add_argument('--classifier', type=str, choices=['DT', 'RF', 'LR', 'XGB', 'GNB', 'ET', 'KNN', 'SVC', 'ALL'], 
                        default='DT', help='Classifier to use (default: DT). Use "ALL" to run all classifiers.')
    parser.add_argument('--output', type=str, default='best_params.txt', help='Output file to save best parameters and AUROC (default: best_params.txt).')
    args = parser.parse_args()

    # Load data
    file_path = args.file
    X, y = load_data(file_path)

    # Get classifiers and their corresponding parameter grids
    classifiers = get_classifiers_and_param_grids()

    # If user chooses "ALL", run for all classifiers, otherwise run for the specified classifier
    if args.classifier == "ALL":
        selected_classifiers = classifiers
    else:
        selected_classifiers = {args.classifier: classifiers.get(args.classifier)}

        if selected_classifiers[args.classifier] is None:
            print(f"Error: Classifier {args.classifier} is not valid. Please choose from {list(classifiers.keys())} or 'ALL'.")
            exit(1)

    # Perform grid search and calculate AUROC
    all_results = []
    for clf_name, (clf, param_grid) in selected_classifiers.items():
        results = perform_grid_search(X, y, clf_name, clf, param_grid)
        all_results.extend(results)

    # Save best parameters and AUROC
    output_file = args.output
    save_best_params(all_results, output_file)

    print(f"Best parameters and AUROC saved to {output_file}")

