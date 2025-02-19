import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from heart_attack_predictor_v2 import HeartAttackRiskPredictor


def analyze_smote_impact():
    print("Analyzing SMOTE Impact on Model Performance...")

    # Initialize two predictors - one with SMOTE and one without
    predictor_with_smote = HeartAttackRiskPredictor('heart_attack_risk_dataset.csv')
    predictor_no_smote = HeartAttackRiskPredictor('heart_attack_risk_dataset.csv')

    # Process data for both predictors
    predictor_with_smote.load_and_preprocess_data()
    predictor_no_smote.load_and_preprocess_data()

    # Apply SMOTE only to one predictor
    predictor_with_smote.balance_data()

    # Train models for both scenarios
    print("\nTraining models without SMOTE...")
    predictor_no_smote.train_models()

    print("\nTraining models with SMOTE...")
    predictor_with_smote.train_models()

    # Collect results
    models = list(predictor_with_smote.models.keys())
    accuracies_with_smote = [predictor_with_smote.model_performances[model]['accuracy']
                             for model in models]
    accuracies_no_smote = [predictor_no_smote.model_performances[model]['accuracy']
                           for model in models]

    # Create comparison plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, accuracies_no_smote, width, label='Without SMOTE', color='lightcoral')
    plt.bar(x + width / 2, accuracies_with_smote, width, label='With SMOTE', color='lightgreen')

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison: With vs Without SMOTE')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/smote_comparison.png')
    plt.close()

    # Print detailed comparison
    print("\nDetailed Performance Comparison:")
    print("\nModel Performance Differences (SMOTE - No SMOTE):")
    for i, model in enumerate(models):
        diff = accuracies_with_smote[i] - accuracies_no_smote[i]
        print(f"{model:15} : {diff:+.4f}")

    # Class distribution analysis
    print("\nClass Distribution Analysis:")
    print("\nWithout SMOTE:")
    print("Training set distribution:", np.bincount(predictor_no_smote.y_train))
    print("\nWith SMOTE:")
    print("Training set distribution:", np.bincount(predictor_with_smote.y_train))

    # Create class distribution plot
    plt.figure(figsize=(10, 5))

    # Plot without SMOTE
    plt.subplot(1, 2, 1)
    sns.countplot(x=predictor_no_smote.y_train)
    plt.title('Class Distribution\nWithout SMOTE')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')

    # Plot with SMOTE
    plt.subplot(1, 2, 2)
    sns.countplot(x=predictor_with_smote.y_train)
    plt.title('Class Distribution\nWith SMOTE')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig('images/class_distribution_comparison.png')
    plt.close()

    return predictor_with_smote, predictor_no_smote


if __name__ == "__main__":
    predictor_with_smote, predictor_no_smote = analyze_smote_impact()