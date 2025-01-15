'''
Utils for working with data, model predictions and evaluating model output.

requirements:
    - pandas
    - numpy
    - sklearn
'''

import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import requests
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup


"""
Prints out evaluation of a machine learning model's predictions on a test dataset.

Input:
    - model_name(str): machine learning model name
    - params_list(array of str): list of model parameters
    - X_test(pd.DataFrame): test features used for prediction
    - y_test(array): test labels
    - y_pred(array): predictions

Output:
    - MSE, R-squared and Adjusted R-squared for the model
"""
def model_eval(model_name, params_list, X_test, y_test, y_pred):
    print(f"`Model name:` {model_name}")

    print("\n`Model parameters:`\n")
    for param in params_list:
        print(param)

    print("\n`Features:`\n")
    for col in X_test.columns:
        print(col)

    # Evaluating the dataset
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'\n- Mean Squared Error: {round(mse, 5)}')
    print(f'- R-squared: {round(r2, 5)}')

    # Calculate Adjusted R-squared
    n = X_test.shape[0]
    p = X_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    print(f'- Adjusted R-squared: {round(adjusted_r2, 5)}')


"""
Fixes predicted values to be in the range from 0 to 1.
Sets a value as 1 if it's greater than 1.
Sets a value as 0 if it's negative.

Input: an array of prediction values
Output: numpy array with predictions in range 0-1
"""
def fix_pred_range(y_pred):
    return np.array([1 if i > 1 else 0 if i < 0 else i for i in y_pred])


"""
Checks violations of transitive consistency rule.

Input:
    - y_pred: pandas dataframe consisting of two columns: id, pred

    Example of y_pred:
    id  pred
    1   0.12345678901
    8   0.12345678901
    13  0.12345678901

    - df_pred: pandas dataframe used to train the model that produced y_pred

Output:
    - number transitive inconsistencies
    - dataframe with transitive inconsistencies
"""
def transitivity_check(y_pred, df_pred):

    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    transitivity_df = pd.DataFrame({
        'id': np.array(y_pred['id']),
        'y_pred_a': np.array(y_pred['pred']),
        'y_pred_b': 1 - np.array(y_pred['pred']),
        'transitivity': 'unchecked'
    })

    transitivity_df = pd.merge(
        transitivity_df,
        df_pred[['id', 'project_a', 'project_b']],
        on='id', how='left'
    ).set_index('id')

    transitivity_df['compare'] = np.where(
        transitivity_df['y_pred_a'] < transitivity_df['y_pred_b'], '<', '>'
    )

    for ind, row in transitivity_df.iterrows():
        if row['y_pred_a'] < row['y_pred_b']:
            a_weight, a_url, b_weight, b_url = row['y_pred_a'], row['project_a'], \
              row['y_pred_b'], row['project_b']
        else:
            a_weight, a_url, b_weight, b_url = row['y_pred_b'], row['project_b'], \
              row['y_pred_a'], row['project_a']

        df_check = transitivity_df[
            (transitivity_df['project_a'] == b_url) |
            (transitivity_df['project_b'] == b_url)
        ]

        df_check['transitivity'] = np.where(
            df_check['project_a'] == b_url,
            np.where(
                df_check['compare'] == '>',
                'checked',
                np.where(
                    a_weight < df_check['y_pred_b'],
                    'checked',
                    'violated'
                )
            ),
            np.where(
                df_check['compare'] == '<',
                'checked',
                np.where(
                    a_weight < df_check['y_pred_a'],
                    'checked',
                    'violated'
                )
            )
        )

        transitivity_df.update(df_check)

    unchecked_violated_df = transitivity_df[
            (transitivity_df['transitivity'] == 'unchecked') |
            (transitivity_df['transitivity'] == 'violated')
    ]

    if unchecked_violated_df.shape[0] == 0:
        print("Number of transitive inconsistencies or unchecked cases: 0")
    else:
        print(
            f'''Number of transitive inconsistencies or unchecked cases: {unchecked_violated_df.shape[0]}''',
            f'''\n{'-'*130}\nDataframe with transitive inconsistencies:\n{'-'*130}\n'''
        )
        return transitivity_df


"""
Encodes both column a and b as labels

Input:
    - column_a(str): column name for project a
    - column_b(str): column name for project b
    - df(pd.DataFrame): dataframe where need to label values
"""
def label_encode_ab(column_a, column_b, df):
    ab_column = pd.concat([df[f'{column_a}'], df[f'{column_b}']]).unique()

    label_encoder = LabelEncoder()

    ab_labels = label_encoder.fit_transform(ab_column)

    mapping = dict(zip(ab_column, ab_labels))

    df[f'{column_a}'] = df[f'{column_a}'].map(mapping)
    df[f'{column_b}'] = df[f'{column_b}'].map(mapping)


"""
Function to fetch README content from a GitHub repository by url.

Input:
    - url_arr(array): an array with github repository urls

Output:
    - list of tuples [(url, description)] with README content
"""
def get_github_readme(url_arr):
    parser = MarkdownIt()

    def fetch(url):
        try:
            if (requests.get(url + '/blob/main/README.md') != 200):
              response = requests.get(url + '/blob/master/README.md')
            else:
              response = requests.get(url + '/blob/main/README.md')
            soup = BeautifulSoup(response.text, 'html.parser')
            readme_content = soup.find('article').get_text() if soup.find('article') else "No README found."
            text_readme = parser.render(readme_content)
            return readme_content
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            return "Unsuccessful request"

    descriptions = [(url, fetch(url)) for url in url_arr]

    for i in range(len(descriptions)):
        modified_description = descriptions[i][1].replace("\n", " ")
        modified_tuple = (descriptions[i][0], modified_description)
        descriptions[i] = modified_tuple

    return descriptions


"""
Makes triplets for consistency_check function.

Input:
    - pred_df(pd.DataFrame): id, pred columns
    - df_train(pd.DataFrame): df used to train model

Output:
    - triplets: triplets = [(0,1,5), (1,2,3), (0,2,8)]

"""
def make_triplets(pred_df, df_train):
    pred_df['pred_b'] = 1 - pred_df['pred']
    pred_df = pred_df.rename(columns={"pred": "pred_a"})
    pred_df['diff'] = pred_df['pred_a'] - pred_df['pred_b']

    df = pd.merge(
        pred_df[['id', 'diff']],
        df_train[['id', 'project_a', 'project_b']],
        on='id', how='left'
    )

    label_encode_ab('project_a', 'project_b', df)
    df = df.drop(columns={'id', })
    triplets = [(row[1][1], row[1][2], row[1][0]) for row in df.iterrows()]
    return triplets
