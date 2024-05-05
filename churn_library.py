"""
This script is designed to process and analyze customer data for churn prediction.
It includes functions for importing data, performing exploratory data analysis (EDA),
encoding categorical features, feature engineering, model training, evaluation, and visualization.
It uses various machine learning techniques,including logistic regression
and random forest classifiers,
and employs grid search for hyperparameter tuning. The script is structured to save plots,
classification reports, and trained models, facilitating a comprehensive analysis workflow.

Dependencies:
    - pandas: For data manipulation and analysis.
    - numpy: For numerical operations.
    - matplotlib and seaborn: For plotting graphs.
    - scikit-learn: For machine learning models and evaluation metrics.
    - joblib: For model serialization.

Author: Joel Orellana
Creation Date: 04/05/2024
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_extracted = pd.read_csv(pth, index_col=0)
    return df_extracted


def perform_eda(df_eda):
    '''
    perform eda on df and save figures to images folder
    input:
            df_eda: pandas dataframe

    output:
            None
    '''
    print("First five columns of df: ")
    print(df_eda.head())
    print(f"The shape of the df is {df_eda.shape}")
    print("Are there NaN values?")
    print(df_eda.isnull().sum())
    print("Summary Stats")
    print(df_eda.describe())
    print("Creating graphs...")
    df_eda['Churn'] = df_eda['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
        )
    plt.figure(figsize=(20,10))
    df_eda['Churn'].hist()
    plt.title('Distribution of Churn')
    plt.xlabel('Churn')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('images/eda/churn_histogram.png', dpi=300)
    plt.close()

    plt.figure(figsize=(20,10))
    df_eda['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.xlabel('Customer Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('images/eda/customer_age_histogram.png', dpi=300)
    plt.close()
    plt.figure(figsize=(20,10))
    df_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Normalized Distribution of Marital Status')  # Title of the plot
    plt.xlabel('Marital Status')  # X-axis label
    plt.ylabel('Proportion')  # Y-axis label
    plt.tight_layout()
    plt.savefig('images/eda/marital_status_bar.png', dpi=300)  # Save the figure
    plt.close()  # Close the figure

    plt.figure(figsize=(20,10))
    # Show distributions of 'Total_Trans_Ct' and
    # add a smooth curve obtained using a kernel density estimate
    sns.histplot(df_eda['Total_Trans_Ct'], stat='density', kde=True)
    # Add titles and labels
    plt.title('Density Plot with Histogram of Total Transaction Count')
    plt.xlabel('Total Transaction Count')
    plt.ylabel('Density')
    plt.tight_layout()
    # Save the figure
    plt.savefig('images/eda/total_trans_ct_density_histogram.png', dpi=300)
    # Close the plot to free up memory
    plt.close()

    plt.figure(figsize=(20,10))
    sns.heatmap(df_eda.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.tight_layout()
    plt.savefig('images/eda/correlation_heatmap.png', dpi=300)
    plt.close()
    print("Graphs saved.")

def encoder_helper(df_encode, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df_encode: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            df_encode: pandas dataframe with new columns for
    '''
    print("Encoding categorical columns...")
    df_encode[response] = df_encode['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    df_encode.drop(["Attrition_Flag", "CLIENTNUM"], axis=1, inplace=True)
    for cat in category_lst:
        cat_groups = df_encode.groupby(cat).mean()[response]
        df_encode[f'{cat}_{response}'] = df_encode[cat].map(cat_groups)
    # removing non processed columns
    df_encode.drop(category_lst, axis=1, inplace=True)
    print("Successful!")

    return df_encode



def perform_feature_engineering(df_fe, category_lst, response='Churn'):
    '''
    input:
              df_fe: pandas dataframe
              category_lst: list of columns that contain categorical features
              response: string of response name [argument that could be used
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    print("Performing feature engineering...")
    df_fe = encoder_helper(df_fe, category_lst, response)
    y = df_fe[response]
    print(df_fe.head())
    X = df_fe.drop(response, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    print("Feature engineering succeed!")
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    print("Saving the img report...")
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.0, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
                         {'fontsize': 10}, fontproperties = 'monospace'
                            ) # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
                           {'fontsize': 10}, fontproperties = 'monospace'
                            ) # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/rf_classification_report.png', dpi=300)
    plt.close()

    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.0, str('Logistic Regression Train'),
                           {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
                           {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('images/results/lr_classification_report.png', dpi=300)
    plt.close()
    print("Saved Classification reports!")



def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    print("Creating plot of feature importances...")
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_pth}/feature_importance.png", dpi=300)
    plt.close()
    print(f"Saved at {output_pth}")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    print("Training started...!")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    print("Creating plot of model results...!")

    # plots
    plt.figure(figsize=(15, 8))
    ax_0 = plt.gca()
    plot_roc_curve(lrc, X_test, y_test, ax=ax_0, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax_0, alpha=0.8)
    plt.tight_layout()
    plt.savefig("images/results/roc_curve.png", dpi=300)
    plt.close()

    print("Plots saved!")
    # creating classification report
    # for random forest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # save best model
    best_rfc_model = cv_rfc.best_estimator_
    joblib.dump(best_rfc_model, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    feature_importance_plot(best_rfc_model, X_train, "images/results")
    print("finished...!")


if __name__ == "__main__":
    DATA_PATH = "data/bank_data.csv"
    cat_columns = ['Gender',
                   'Education_Level',
                   'Marital_Status',
                   'Income_Category',
                   'Card_Category']
    df = import_data(DATA_PATH)
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df,
                                                                   cat_columns,
                                                                   response='Churn')
    train_models(X_train, X_test, y_train, y_test)
