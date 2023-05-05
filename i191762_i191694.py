import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import operator
import random                                  
from math import exp 
import numpy.random as npr
from collections import Counter
from itertools import compress
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

def data_preprocessing():
    df = pd.read_excel('./Training_Data.xlsx')
    for column in df.columns:
        if (df[column].isna().sum() > 0):
            counts = dict(df[column].value_counts())
            if (len(counts.keys()) > 1):
                df[column] = df[column].replace(np.nan, df[column].mode()[0])
            elif (len(counts.keys()) == 1):
                rep = 'not_' + list(counts.keys())[0]
                df[column] = df[column].replace(np.nan, rep)
    
    df = df.astype({"EncounterId": str})
    le = LabelEncoder()
    for column in df.columns:
        if (df[column].dtype == 'object'):
            df[column] = le.fit_transform(df[column])
    
    df.to_csv('./preprocessed.csv', index=False)

def model_training_and_eval():
    df = pd.read_csv('./preprocessed.csv')
    x = df.drop(columns = ['ReadmissionWithin_90Days'])
    y = df['ReadmissionWithin_90Days']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Model_Evaluation")

    max_depth = [10, 20]
    estimators = [50, 100]
    count = 1
    for depth in max_depth:
        for est in estimators:
            name = 'Model_Run_' + str(count)
            with mlflow.start_run(run_name = name, description = 'Model Evaluation Runs'):
                rfc = RandomForestClassifier()
                rfc.fit(x_train, y_train)
                y_pred = rfc.predict(x_test)
                score =  accuracy_score(y_pred, y_test)*100
                mlflow.log_metric('accuracy', score)
                count += 1


def model_selection_and_deployment():
    exp_name = 'Model_Evaluation'
    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    all_runs = mlflow.search_runs(experiment_ids=exp_id, order_by=['metrics.accuracy'])
    best_run = all_runs.loc[all_runs['metrics.accuracy'].idxmin()]
    best_run_id = best_run['run_id']
    path = f"runs:/{best_run_id}/model"
    load_model = mlflow.sklearn.load_model(path)
    mlflow.register_model(model_uri=path, name="Production_Model")


airflow_args = {
    'owner':'razi_abdur',
    'depends_on_past':False,
    'start_date': datetime(2023, 5, 5),
    'retries':1,
    'retry_delay':timedelta(minutes=2) ,
    'user':'admin',
    'password':'admin'
}

dag = DAG(
    'ML_Pipeline',
    default_args=airflow_args,
    description='End-to-end ML piple example',
    schedule=timedelta(days=1)
)

with dag:
    preprocessing_module = PythonOperator(task_id='preprocess_data', python_callable=data_preprocessing)
    model_train_and_eval_module = PythonOperator(task_id='model_train_and_eval', python_callable=model_training_and_eval)
    model_select_and_deploy_module = PythonOperator(task_id='model_selection_and_deployment', python_callable=model_selection_and_deployment)

    #task order
    preprocessing_module >> model_train_and_eval_module >> model_select_and_deploy_module