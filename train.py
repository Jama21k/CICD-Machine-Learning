import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import skops.io as sio 
import os

def main():
    print("Loading data...")
    df = pd.read_csv("Data/drug200.csv")

    drug_df = df.sample(frac=1)
    drug_df.head(3)


    X = drug_df.drop("Drug", axis=1).values
    y = drug_df.Drug.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)
    
    # Placeholder for training logic
    print("Training model...")
    cat_col = [1,2,3]
    num_col = [0,4]
    transform = ColumnTransformer([("encoder", OrdinalEncoder(), cat_col),
                               ("num_imputer", SimpleImputer(strategy="median"), num_col ),
                               ("num_scaler", StandardScaler(), num_col)])
    pipeline = Pipeline(
        steps=[("preporcessing", transform),
           ("model", RandomForestClassifier(n_estimators=100, random_state=125))])

    pipeline.fit(X_train, y_train)


    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print("Accuracy:", str(round(accuracy, 2)*100)+"%", "F1:", round(f1, 2))
    cm = confusion_matrix(y_test, predictions, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot()
    plt.savefig("Results/mode_results.png", dpi=120)
    
    # Save artifacts
    os.makedirs("Model", exist_ok=True)
    os.makedirs("Results", exist_ok=True)
    
    # joblib.dump(model, "Model/drug_model.pkl")
    print("Training complete!")
    sio.dump(pipeline, "Model/drug_pipeline.skops")
if __name__ == "__main__":
    main()
