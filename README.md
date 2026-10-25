# Titanic Survival Prediction

## Project overview

This project uses passenger information from the Titanic manifest to predict survival (`Survived`: 0 = did not survive, 1 = survived). The notebook explores the data, engineers family and title features, and compares Logistic Regression, Random Forest, and Gaussian Naive Bayes on a reproducible, stratified 80/20 train-test split.

Preprocessing is part of each scikit-learn pipeline, so imputers, encoders, and scalers are fitted on training data only. The final model is selected from the measured holdout metrics rather than a predetermined choice.

## Dataset

`Titanic-Dataset.csv` contains 891 passengers and 12 columns: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`. The notebook verifies this schema before modeling. In this copy, `Age` has 177 missing values, `Cabin` has 687, and `Embarked` has 2. The original CSV is read-only from the notebook's perspective; feature engineering is performed on a separate copy.

## Dependencies

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn 1.2 or newer

Install the Python packages in your environment:

```bash
python -m pip install jupyter pandas numpy matplotlib seaborn "scikit-learn>=1.2"
```

## Run the notebook

1. Place `Titanic-Dataset.csv` beside `Titanic_Survival_ML.ipynb`.
2. Install the dependencies listed above.
3. Start Jupyter from this project folder:

   ```bash
   jupyter notebook
   ```

4. Open `Titanic_Survival_ML.ipynb` and choose the installed Python environment.
5. Run all cells from top to bottom. The notebook saves figures in `visualizations/` and creates `Summary_Report.pdf` in this folder after calculating the actual test results.