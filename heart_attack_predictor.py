import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import time
from tqdm import tqdm


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

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=35
        )

        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

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

    def train_models(self):
        """Train different models and store their performances."""
        print("\nTraining models...")

        # Initialize models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=315),
            'SVM': SVC(kernel='rbf', random_state=456),
            'AdaBoost': AdaBoostClassifier(random_state=46),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }

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
            # Calculate classification report with zero_division parameter
            class_report = classification_report(
                self.y_test,
                y_pred,
                zero_division=1,  # Handle zero-division case
                output_dict=True  # Get the report as a dictionary for easier handling
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
            # Convert dict back to formatted string for display
            report_str = ""
            for label in class_report.keys():
                if isinstance(class_report[label], dict):
                    report_str += f"\n{label:>15} "
                    for metric, value in class_report[label].items():
                        if isinstance(value, float):
                            report_str += f"{metric}: {value:.3f} "
            print(report_str)

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
        # Apply PCA
        pca = PCA()
        X_pca = pca.fit_transform(self.X_train)

        # Calculate explained variance ratio
        exp_var_ratio = pca.explained_variance_ratio_
        cum_exp_var_ratio = np.cumsum(exp_var_ratio)

        # Plot explained variance ratio
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(exp_var_ratio) + 1), cum_exp_var_ratio, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Analysis - Explained Variance Ratio')
        plt.grid(True)
        plt.savefig('images/pca_analysis.png')
        plt.close()

        return cum_exp_var_ratio


# Example usage
def main():
    print("Starting Heart Attack Risk Prediction Analysis...")

    # Initialize predictor
    predictor = HeartAttackRiskPredictor('heart_attack_risk_dataset.csv')

    # Load and preprocess data
    predictor.load_and_preprocess_data()

    # Perform EDA
    predictor.perform_eda()

    # Train models
    predictor.train_models()

    # Evaluate models
    predictor.evaluate_models()

    # Perform dimension reduction analysis
    predictor.dimension_reduction_analysis()


if __name__ == "__main__":
    main()