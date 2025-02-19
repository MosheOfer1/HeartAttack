import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Machine Learning Models and Tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# SMOTE for balancing classes
from imblearn.over_sampling import SMOTE


class HeartAttackRiskPredictor:
    def __init__(self, data_path):
        """Initialize the predictor with data path."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_performances = {}

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        print("Loading and preprocessing data...")

        # Load data
        self.df = pd.read_csv(self.data_path)

        # Convert categorical variables to numeric
        le = LabelEncoder()
        categorical_columns = ['Gender', 'Physical_Activity_Level', 'Stress_Level',
                               'Chest_Pain_Type', 'Thalassemia', 'ECG_Results',
                               'Heart_Attack_Risk']

        for col in tqdm(categorical_columns, desc="Encoding categorical variables"):
            self.df[col] = le.fit_transform(self.df[col])

        # Separate features and target
        self.X = self.df.drop('Heart_Attack_Risk', axis=1)
        self.y = self.df['Heart_Attack_Risk']

        # Split the data (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=35
        )

        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        return self

    def balance_data(self):
        """Balance training data using SMOTE."""
        print("\nBalancing training data with SMOTE...")
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print("New class distribution:", np.bincount(self.y_train))
        return self

    def perform_eda(self):
        """Perform Exploratory Data Analysis."""
        print("\nPerforming Exploratory Data Analysis...")
        # Create correlation matrix
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Features')
        plt.tight_layout()
        plt.savefig('images/correlation_matrix.png')
        plt.close()

        # Distribution of target variable
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x='Heart_Attack_Risk')
        plt.title('Distribution of Heart Attack Risk Levels')
        plt.savefig('images/risk_distribution.png')
        plt.close()

        # Feature importance analysis using Decision Tree
        dt = DecisionTreeClassifier(random_state=31)
        dt.fit(self.X_train, self.y_train)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': dt.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features')
        plt.savefig('images/feature_importance.png')
        plt.close()

    def tune_model(self, model, param_grid, model_name):
        """Tune hyperparameters for a given model using GridSearchCV."""
        print(f"\nTuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_models(self):
        """Train different models and store their performances."""
        print("\nTraining models...")

        # Initialize models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=315),
            'SVM': SVC(kernel='rbf', random_state=456, probability=True),
            'AdaBoost': AdaBoostClassifier(random_state=46),
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'Random Forest': RandomForestClassifier(random_state=123),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123)
        }

        # Hyperparameter tuning for SVM as an example
        svm_param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
        models['SVM'] = self.tune_model(models['SVM'], svm_param_grid, "SVM")

        # Optionally, hyperparameter tuning can be added for other models as well

        # Train and evaluate each model
        for name, model in tqdm(models.items(), desc="Training models"):
            start_time = time.time()

            # Train the model
            model.fit(self.X_train, self.y_train)

            # Make predictions
            y_pred = model.predict(self.X_test)

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            training_time = time.time() - start_time

            # Store results
            self.models[name] = model
            class_report = classification_report(
                self.y_test,
                y_pred,
                zero_division=1,
                output_dict=True
            )

            self.model_performances[name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'classification_report': class_report,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }

            # Print immediate results for this model
            print(f"\nResults for {name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Training time: {training_time:.2f} seconds")
            print("\nClassification Report:")

        return self

    def evaluate_models(self):
        """Compare model performances and create visualizations."""
        print("\nEvaluating model performances...")
        # Prepare performance metrics for visualization
        accuracies = [perf['accuracy'] for perf in self.model_performances.values()]
        training_times = [perf['training_time'] for perf in self.model_performances.values()]

        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(self.models.keys()), y=accuracies)
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/model_accuracy_comparison.png')
        plt.close()

        # Plot training time comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(self.models.keys()), y=training_times)
        plt.title('Model Training Time Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/model_training_time_comparison.png')
        plt.close()

        # Print detailed performance metrics
        for name, performance in self.model_performances.items():
            print(f"\nModel: {name}")
            print(f"Accuracy: {performance['accuracy']:.4f}")
            print(f"Training Time: {performance['training_time']:.2f} seconds")
            print("\nClassification Report:")
            print(performance['classification_report'])

    def dimension_reduction_analysis(self):
        """Perform PCA analysis and visualize results."""
        print("\nPerforming dimension reduction analysis...")
        from sklearn.decomposition import PCA
        pca = PCA()
        X_pca = pca.fit_transform(self.X_train)
        exp_var_ratio = pca.explained_variance_ratio_
        cum_exp_var_ratio = np.cumsum(exp_var_ratio)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(exp_var_ratio) + 1), cum_exp_var_ratio, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Analysis - Explained Variance Ratio')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('images/pca_analysis.png')
        plt.close()

        return cum_exp_var_ratio


def visualize_decision_tree(predictor, max_depth=3):
    """
    Visualize a single decision tree from the predictor's Decision Tree model.

    Parameters:
    -----------
    predictor : HeartAttackRiskPredictor
        The predictor instance containing the trained Decision Tree model
    max_depth : int, optional (default=3)
        Maximum depth of the tree to visualize for clarity

    Returns:
    --------
    None : Saves the tree visualization to 'images/decision_tree.png'
    """
    import graphviz
    from sklearn.tree import export_graphviz
    import os

    # Get the trained decision tree model
    if 'Decision Tree' not in predictor.models:
        raise ValueError("Decision Tree model not found in predictor's models")

    tree_model = predictor.models['Decision Tree']

    # Create a new decision tree with limited depth for visualization
    tree_model_viz = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=315
    )
    tree_model_viz.fit(predictor.X_train, predictor.y_train)

    # Get feature names
    feature_names = predictor.X.columns.tolist()

    # Define class names
    class_names = ['Low Risk', 'Medium Risk', 'High Risk']

    # Create dot data
    dot_data = export_graphviz(
        tree_model_viz,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth
    )

    # Create graph
    graph = graphviz.Source(dot_data)

    # Ensure images directory exists
    os.makedirs('images', exist_ok=True)

    # Save visualization
    graph.render('images/decision_tree', format='png', cleanup=True)

    print(f"Decision tree visualization saved as 'images/decision_tree.png'")

    # Print feature importance for the visualized tree
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': tree_model_viz.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 most important features in this tree:")
    print(feature_importance.head().to_string())

# # Example usage:
# predictor = HeartAttackRiskPredictor('heart_attack_risk_dataset.csv')
# predictor.load_and_preprocess_data()
# predictor.train_models()
# visualize_decision_tree(predictor, max_depth=3)


# Example usage
def main():
    print("Starting Heart Attack Risk Prediction Analysis...")

    # Initialize predictor with dataset path
    predictor = HeartAttackRiskPredictor('heart_attack_risk_dataset.csv')

    # Load and preprocess data
    predictor.load_and_preprocess_data()

    # Optional: Balance the training data to address class imbalance
    predictor.balance_data()

    # Perform Exploratory Data Analysis
    predictor.perform_eda()

    # Train models (including hyperparameter tuning and additional models)
    predictor.train_models()

    # Evaluate model performances
    predictor.evaluate_models()

    # Perform dimension reduction analysis
    predictor.dimension_reduction_analysis()


if __name__ == "__main__":
    main()
