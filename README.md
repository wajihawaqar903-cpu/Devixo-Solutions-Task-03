# Devixo-Solutions-Task-03
Devixo Solutions AI/ML Internship Task 03 — End-to-End Machine Learning Project for Telco Customer Churn Prediction.
Project Objective
This project builds and compares machine-learning models to predict
whether a telecom customer is likely to leave the service (Churn). The
goal is to practice a complete workflow: preparing data, training
models, comparing results, tuning a model, checking cross-validation,
examining feature importance, and saving a reusable model.
Dataset
- Name: Telco Customer Churn
- Source: Kaggle --- Telco Customer
  Churn
- Raw dataset size: 7,043 rows and 21 columns.
- Target: Churn (Yes / No). After one-hot encoding, the
  positive target may appear as Churn_Yes.
- Examples of features: tenure, contract type, internet service,
  monthly charges, total charges, and payment method.
Technologies and Libraries
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib and Pickle
- Google Colab
-  Project Files
- `Telco customer churn.ipynb`: full notebook
- `Telco_Customer_Churn_Report.pdf`: project report
- `requirements.txt`: libraries needed
- `best_model.joblib` / `.pkl`: saved best model
- `scaler.joblib`, `column_names.joblib` / `.pkl`: saved scaler and feature columns
- `*.png`: charts (model comparison, feature importance, tuned model metrics)
- streamlit Web Application
Project Workflow
1. Load and inspect data: read the CSV and review the first rows,
   dimensions, column types, and summary statistics.
2. Explore the data: chart churn counts, churn by contract, churn
   by internet service, monthly charges by churn, and numeric
   correlations.
3. Handle missing values: convert TotalCharges to numeric; fill
   missing numeric values with medians and categorical values with
   modes.
4. Remove duplicates: check for duplicate rows and remove them if
   present.
5. Feature engineering: remove the identifier customerID; create
   TenureGroup; create the ChargesPerTenureMonth feature from
   monthly charges and tenure.
6. Encode categorical columns: use one-hot encoding to make
   categorical data usable by scikit-learn.
7. Feature selection: use SelectKBest with the ANOVA F-test to
   retain up to 15 features.
8. Split and scale: split the data into training and test sets and
   scale features with StandardScaler.
9. Train and compare models: Decision Tree, Random Forest, Support
   Vector Classifier (SVC), and Gradient Boosting.
10. Tune the model: use RandomizedSearchCV on Random Forest with 6
    sampled settings and 3-fold cross-validation, optimizing F1-score.
11. Cross-validation: 5-fold `cross_val_score` on the final model gave
    scores of [0.xx, 0.xx, 0.xx, 0.xx, 0.xx] with a mean of 0.xx.
12. Feature importance: plot the sorted feature_importances_
    values from the tuned Random Forest.
13. Save and test: save the model using Joblib and Pickle, save the
    scaler and feature-column names, reload the files, and make a test
    prediction.
Model Results
Results on the held-out test set. Baseline scores are rounded as shown
in the notebook; ROC-AUC is reported for the tuned model.
  Model          Accuracy    Precision       Recall   F1 (Churn)      ROC-AUC
                               (Churn)      (Churn)              
  Decision         0.7275         0.49         0.49         0.49          ---
  Tree                                                           

  Random           0.7622         0.56         0.47         0.51          ---
  Forest                                                         

  SVC              0.7260         0.49         0.81         0.61          ---

  Gradient         0.7928         0.64         0.50         0.56          ---
  Boosting                                                       
  Tuned      0.7630   0.5403   0.7166   0.6161   0.8259
  Random
  Forest                                                       
Hyperparameter tuning result
- Best parameters: n_estimators=50, max_depth=10,
  min_samples_split=2, min_samples_leaf=1
- Best 3-fold tuning CV F1-score: 0.6113
- Tuned Random Forest test accuracy: 0.7630
- Tuned Random Forest test precision: 0.5403
- Tuned Random Forest test recall: 0.7166
- Tuned Random Forest test F1-score: 0.6161
- Tuned Random Forest test ROC-AUC: 0.8259
The tuned Random Forest was selected as the final model because its
churn-class F1-score was the strongest among the reported models and it
achieved higher churn recall than the original Random Forest. Gradient
Boosting had the highest reported accuracy, but its churn recall was
lower, meaning it missed more customers who actually churned. The model
choice therefore prioritizes identifying churners rather than accuracy
alone.
Limitations
- The classes are imbalanced: the dataset contains more non-churn
  customers than churn customers.
- The project uses one public dataset, so results may not generalize
  to every telecom provider.
- Hyperparameter tuning was limited to six sampled settings and 3-fold
  CV to keep runtime manageable.
- In the current notebook workflow, feature selection and some
  preprocessing were performed before the train/test split. This can
  cause data leakage and make evaluation scores optimistic. For a
  stronger evaluation, split first and fit imputation, feature
  selection, and scaling only on training folds using a scikit-learn
  pipeline.
Future Improvements
- Use more recent customer records and test the model on data from
  another time period or provider.
- Tune more hyperparameter combinations with a controlled search
  budget.
- Compare with XGBoost or other gradient-boosting implementations.
- Build a leakage-safe preprocessing and modeling pipeline.
- Examine precision-recall curves and choose a threshold based on the
  cost of missed churners versus unnecessary retention offers.
- Deploy a simple prediction interface with Streamlit and document how
  to use it.
How to Run
1. Open the notebook in Google Colab.
2. Upload the Telco Customer Churn CSV from Kaggle.
3. Run the notebook cells in order, including preprocessing, model
   training, tuning, cross-validation, feature importance, and
   saving/loading.
4. Keep best_model.joblib or best_model.pkl, scaler.joblib, and
   the saved feature-column names together when using the model for new
   predictions.
5. Apply the same preprocessing, column order, and fitted scaler to new
   data before prediction.
Saved Artifacts
- best_model.joblib
- best_model.pkl
- scaler.joblib
- column_names.joblib
- column_names.pkl
Conclusion
This project demonstrates an end-to-end churn prediction workflow. The
tuned Random Forest provides a useful balance between churn detection
and overall performance, but the results should be re-evaluated after
moving all preprocessing and feature selection inside a cross-validation
pipeline.
## Streamlit Web Application

A web application was developed using Streamlit to demonstrate the trained machine learning model. It allows users to enter customer details and receive a churn prediction along with the estimated probability of churn.

**Application features:**
- Accepts customer information through an interactive interface.
- Uses the trained Random Forest model to predict customer churn.
- Displays the predicted churn probability.
- Uses the saved model, scaler, and feature column names.

**Technologies Used:** Python, Streamlit, Pandas, Scikit-learn, and Joblib.

**How to Run the Application:**

1. Install the required libraries:
   ```bash
   python -m pip install streamlit pandas scikit-learn joblib
   ```
2. Open a terminal in the folder containing `app.py` and the saved model files.
3. Run:
   ```bash
   python -m streamlit run app.py
   ```

The application opens in a web browser where users can enter customer details and view predictions.

Author:
Wajiha Waqar
domain:AI/ML
ML Task no :3
