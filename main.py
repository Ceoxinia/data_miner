from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from itertools import combinations
from tabulate import tabulate
import json

import pandas as pd
import os


df1 = pd.read_csv('Dataset1.csv')
df2 = pd.read_csv('Dataset2.csv')
df3 = pd.read_excel('Dataset3.xlsx')
#==========================Pretraitement===============================#
def get_column_description(dataset):
    colonnes_description = []
    for column in dataset.columns:
        colonnes_description.append([
            column,
            dataset[column].count(),
            str(dataset.dtypes[column]),
            len(dataset[column].unique())
        ])
    column_description_df = pd.DataFrame(colonnes_description, columns=["Nom", "Valeur non null", "Type", "Nombre de valeur unique"])
    return column_description_df

def clean_dataset(datasetselected, missing_strategy, outlier_strategy):
    # Handling missing values and outliers for each numerical column
    dataset=datasetselected.copy()
    numeric_columns = dataset.select_dtypes(include=['number']).columns

    for column in numeric_columns:
        # Handling missing values
        if missing_strategy == 'mean':
            dataset[column] = dataset[column].fillna(dataset[column].mean())
        elif missing_strategy == 'median':
            dataset[column] = dataset[column].fillna(dataset[column].median())
        elif missing_strategy == 'mode':
            dataset[column] = dataset[column].fillna(dataset[column].mode().iloc[0])

        # Handling outliers
        if outlier_strategy in ['mean', 'median', 'mode']:
            # Calculate the Interquartile Range (IQR)
            Q1 = dataset[column].quantile(0.25)
            Q3 = dataset[column].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate the upper and lower fences
            upper_fence = Q3 + 1.5 * IQR
            lower_fence = Q1 - 1.5 * IQR

            # Replace outliers with mean, median, or mode
            if outlier_strategy == 'mean':
                dataset[column] = dataset[column].apply(lambda x: dataset[column].mean() if x > upper_fence or x < lower_fence else x)
            elif outlier_strategy == 'median':
                dataset[column] = dataset[column].apply(lambda x: dataset[column].median() if x > upper_fence or x < lower_fence else x)
            elif outlier_strategy == 'mode':
                mode_val = dataset[column].mode().iloc[0]
                dataset[column] = dataset[column].apply(lambda x: mode_val if x > upper_fence or x < lower_fence else x)
    return dataset
def min_max_scaling(column):
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val)
    return scaled_column

def z_score_normalization(column):
    mean_val = column.mean()
    std_dev = column.std()
    normalized_column = (column - mean_val) / std_dev
    return normalized_column
def normaliser(datasetcleaned,method):
    dataset = datasetcleaned.copy()
    numeric_columns = dataset.select_dtypes(include='number').columns
    if method.lower() == 'min-max':
        # Using Min-Max scaling for each column
        for column in numeric_columns:
            dataset[column] = min_max_scaling(dataset[column])

    elif method.lower() == 'z-score':
        # Using Z-score normalization for each column
        for column in numeric_columns:
            dataset[column] = z_score_normalization(dataset[column])

    else:
        raise ValueError("Invalid normalization method. Please choose 'min-max' or 'z-score'.")

    return dataset
def discretize(df, colonne, method):
    if method == 'equal_width':
        return discretize_equal_width(df[colonne], k=10)  # Set k as needed
    elif method == 'equal_frequency':
        num_quantiles = int(np.sqrt(len(df)))
        return discretize_equal_frequency(df, colonne, num_quantiles)
    else:
        raise ValueError("Invalid discretization method. Supported methods are 'equal_width' and 'equal_frequency'.")

def discretize_equal_width(col, k):
    min_value = min(col)
    max_value = max(col)
    largeur = (max_value - min_value) / k
    print("Largeur: ", largeur)
    intervals = [min_value + i * largeur for i in range(k)]
    print("Intervals: ", intervals)
    discretized_data = []
    for value in col:
        # Replace commas with dots and convert to float
        value = float(value.replace(',', '.')) if isinstance(value, str) else value
        category = 0
        for i in range(1, len(intervals)):
            if value <= intervals[i]:
                category = i - 1
                break
        discretized_data.append(category)
    return discretized_data

def discretize_equal_frequency(df, colonne, num_quantiles):
    # Replace commas with dots and convert to float
    df[colonne] = df[colonne].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

    # Calculate the position of each quantile
    quantile_positions = [int(len(df) * i / num_quantiles) for i in range(1, num_quantiles)]

    # Sort the values to assign them to the quantiles
    sorted_values = sorted(df[colonne])

    # Initialize a list to store the intervals
    intervals = [float('-inf')] + [sorted_values[pos] for pos in quantile_positions] + [float('inf')]

    # Label each value with the index of the quantile to which it belongs
    df[colonne + '_DEf'] = pd.cut(df[colonne], bins=intervals, labels=False, include_lowest=True)

    return df

def generate_candidates(transactions, k):
              # Create a dictionary to count the support of itemsets
    itemset_counts = {}

    # Iterate through each transaction to count k-itemsets
    for transaction in transactions:
        # Split the transaction string into individual items
        items = transaction.split('_')

        # Use combinations to generate the combinations of k elements
        k_itemsets = list(combinations(items, k))

        # Increment the counter for each itemset in the transaction
        for itemset in k_itemsets:
            if itemset in itemset_counts:
                itemset_counts[itemset] += 1
            else:
                itemset_counts[itemset] = 1

    # Filter candidate itemsets having sufficient support
    candidate_itemsets = [itemset for itemset, count in itemset_counts.items() if count >= k]
    return candidate_itemsets
def calculate_support(transactions, itemsets):
              # Compte le nombre d'occurrences de chaque itemset dans les transactions
    support_counts = {}

    for itemset in itemsets:
        for transaction in transactions:
            if all(item in transaction for item in itemset):
                if itemset in support_counts:
                    support_counts[itemset] += 1
                else:
                    support_counts[itemset] = 1

    return support_counts
def generate_frequent_itemsets(support_counts, k, min_support):
    frequent_itemsets = [itemset for itemset, support in support_counts.items() if support >= min_support]
    return frequent_itemsets

def apriori_algorithm(transactions, min_support):
    k = 1
    frequent_itemsets = []

    frequent_itemsets_k = True

    while frequent_itemsets_k:

        # Génère les k-itemsets candidats Ck
        candidates = generate_candidates(transactions, k)
        #print('-----------------------------------------------------')
        #print("C:", candidates)
        # Calcule le support de chaque candidat
        support_counts = calculate_support(transactions, candidates)
        frequent_itemsets_k = generate_frequent_itemsets(support_counts, k, min_support)
        #print(support_counts)

        frequent_itemsets.extend(frequent_itemsets_k)
        #print("L:",frequent_itemsets)
        #print('-----------------------------------------------------')
        k += 1
    return frequent_itemsets

# Fonction de calcul de la confiance
def calculate_confidence(antecedent, consequent,transactions):

    # Compter le nombre de transactions supportant l'ensemble d'items de la règle
    rule_support = sum(1 for transaction in transactions if set(antecedent).issubset(transaction.split('_')) and set(consequent).issubset(transaction.split('_')))

    # Compter le nombre de transactions supportant l'ensemble d'items du côté gauche de la règle
    antecedent_support = sum(1 for transaction in transactions if set(antecedent).issubset(transaction.split('_')))

    # Éviter une division par zéro
    if antecedent_support == 0:
        return 0.0

    # Calculer la confiance
    confidence = rule_support / antecedent_support
    return confidence
def generate_association_rules(Lk, min_conf, transactions):
    association_rules = []
    for itemset in Lk:
        itemset = set(itemset)
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = set(antecedent)
                    consequent = itemset - antecedent
                    f = calculate_confidence(antecedent, consequent, transactions)
                    if f >= min_conf:
                        # Convert sets to lists before adding to association_rules
                        association_rules.append((list(antecedent), list(consequent)))
    return association_rules


#===========================App Layout======================================#

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H1("Data Mining Project",style={'color': 'blue'}),
        
        # Dataset selection dropdown
        dcc.Dropdown(
            id='data-dropdown',
            options=[
                {'label': 'Dataset 1', 'value': 'df1'},
                {'label': 'Dataset 2', 'value': 'df2'},
                {'label': 'Dataset 3', 'value': 'df3'}
            ],
            value='df1',  # default selection
            style={'width': '50%'}
        ),
        
        # Cleaning methods dropdowns and button
        html.Div([
            html.H5('Select Outliers Handling Method'),
            dcc.Dropdown(
                id='outlier-dropdown',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': 'Mode', 'value': 'mode'}
                ],
                value='mean',
                style={'width': '50%'}
            ),
            html.H5('Select Missing Values Handling Method'),
            dcc.Dropdown(
                id='missing-dropdown',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': 'Mode', 'value': 'mode'}
                ],
                value='mean',
                style={'width': '50%'}
            ),
        ]),
        html.Div([
        html.H5('Normalization Method'),
        dcc.Dropdown(
            id='normalization-method-dropdown',
            options=[
                {'label': 'Min-Max', 'value': 'min-max'},
                {'label': 'Z-Score', 'value': 'z-score'}
            ],
            value='min-max',  # Default selection
            style={'width': '50%'}
        ),
        ]),
        html.Div([
        html.H5('Discretization Method For Dataset3'),
        dcc.Dropdown(
            id='discretization-method-dropdown',
            options=[
                {'label': 'Equal Width', 'value': 'equal_width'},
                {'label': 'Equal Frequency', 'value': 'equal_frequency'}
            ],
            value='equal_width',  # Default selection
            style={'width': '50%'}
        ),
    ]),
    ], style={'width': '30%', 'float': 'left', 'background-color': 'lightgrey'}),
    
    html.Div([
        html.Div(id='output-data-upload'),
        html.Div(id='data-summary'),
        dcc.Graph(id='boxplot'),
        dcc.Graph(id='histogram'),
        dcc.Graph(id='correlation-matrix'),  # New graph for correlation matrix
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    html.Div([
        html.H5('Cleaned DataFrame'),
        dash_table.DataTable(id='cleaned-table',
        style_table={'height': '200px', 'overflowY': 'auto'})
    ]),
    
    html.Div([
        dcc.Graph(id='cleaned-boxplot'),
        dcc.Graph(id='cleaned-histogram')
    ]),
    
    html.Div(id='image-section'),
    html.Div([
    html.H5('Normalized DataFrame'),
    dash_table.DataTable(id='normalized-table',
    style_table={'height': '200px', 'overflowY': 'auto'})    
]),
    html.Div([
    html.H5('Descritisized DataFrame'),
    dash_table.DataTable(id='discretization-table',
    style_table={'height': '200px', 'overflowY': 'auto'})    
]),
    html.Div([
    html.H5('Association Rules Table'),
    dash_table.DataTable(
        id='association-rules-table',
        columns=[
            {'name': 'Min Support', 'id': 'Min Support'},
            {'name': 'Min Confidence', 'id': 'Min Confidence'},
            {'name': 'Association Rules', 'id': 'Association Rules'},
        ],
        style_table={'height': '400px', 'overflowY': 'auto'}
    ),
])

])


def get_selected_dataframe(selected_data):
    if selected_data == 'df1':
        return df1
    elif selected_data == 'df2':
        return df2
    elif selected_data == 'df3':
        return df3

@app.callback([Output('output-data-upload', 'children'),
               Output('data-summary', 'children'),
               Output('boxplot', 'figure'),
               Output('histogram', 'figure'),
               Output('correlation-matrix', 'figure'),
               Output('cleaned-table', 'data'),
               Output('cleaned-boxplot', 'figure'),
               Output('cleaned-histogram', 'figure'),
               Output('normalized-table', 'data'),
               Output('image-section', 'children'),
               Output('discretization-table', 'data'),
               Output('association-rules-table', 'data')],  # Add this Output for image section
              [Input('data-dropdown', 'value'),
               Input('discretization-method-dropdown', 'value'),
               ],
              [State('outlier-dropdown', 'value'),
               State('missing-dropdown', 'value'),
               State('normalization-method-dropdown', 'value')])
def update_output(selected_data,selected_discretization_method, selected_outliers, selected_missing,selected_normalization_method):
    df = get_selected_dataframe(selected_data)
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    image_section = html.Div()

    table = html.Div([
        html.H5(f'Selected Dataset: {selected_data}'),
        dash_table.DataTable(
            df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'height': '200px', 'overflowY': 'auto'}
        ),
    ])

    summary = html.Div([
        html.Hr(),
        html.H5('DataFrame Summary'),
        html.Div([
            html.Div([
                html.H6('Description des columns:'),
                dcc.Markdown(f'```\n{get_column_description(df)}\n```')
            ], style={'width': '60%', 'padding': '10px'}),
            html.Div([
                html.H6('Les Tendances Centrales:'),
                dcc.Markdown(f'```\n{df.describe()}\n```')
            ], style={'width': '60%', 'padding': '10px'})
        ], style={'display': 'flex'})
    ])

    boxplot = px.box(df, y=df.select_dtypes(include=['number']).columns)
    boxplot.update_layout(title_text='Boxplot of Numeric Columns')

    histogram = px.histogram(df, x=df.select_dtypes(include=['number']).columns, marginal="rug")
    histogram.update_layout(title_text='Histogram of Numeric Columns')

    # Correlation matrix graph
    correlation_matrix = px.imshow(df_numeric.corr(), labels=dict(x='Columns', y='Columns'), x=df.columns, y=df.columns)
    correlation_matrix.update_layout(title_text='Correlation Matrix')

    cleaned_df = clean_dataset(df, selected_missing, selected_outliers)
    cleaned_table_data = cleaned_df.to_dict('records')
 
           
    cleaned_boxplot = px.box(cleaned_df, y=cleaned_df.select_dtypes(include=['number']).columns)
    cleaned_boxplot.update_layout(title_text='Cleaned Boxplot of Numeric Columns')

    cleaned_histogram = px.histogram(cleaned_df, x=cleaned_df.select_dtypes(include=['number']).columns, marginal="rug")
    cleaned_histogram.update_layout(title_text='Cleaned Histogram of Numeric Columns')
    
    normalized_df = normaliser(cleaned_df, selected_normalization_method)
    normalized_table_data = normalized_df.to_dict('records')
    
    if selected_data == 'df2':
        image_section = html.Div([
        html.H5('Interesting Insights Of Dataset 2:'),
        html.Img(src='/dataset2/confirmedtests.png'),    # ... (other image sources)
        ])
        discretized_table = []
        table_data =[]
    elif selected_data == 'df3':
        descr = df3.copy()
        descr['Temperature_D'] = discretize(df, 'Temperature', selected_discretization_method)
        discretized_table = descr.to_dict('records')
        Dataset2_bis = pd.DataFrame({
        'Transactions': descr.apply(lambda row: f"{row['Temperature_D']}_{row['Crop']}_{row['Fertilizer']}" if all(pd.notna(row[col]) for col in ['Temperature_D', 'Crop','Fertilizer']) else None, axis=1)})
        transactions = Dataset2_bis['Transactions'].tolist()
        frequent_itemsets = apriori_algorithm(transactions, 2)
        association_rules = generate_association_rules(frequent_itemsets, 0, transactions)
        # Créez une liste pour stocker les données du tableau
        min_supp_values = [1, 2, 3]  # nbr d'apparition
        min_conf_values = [0.05, 0.1, 0.5, 1]  # confiance
        results_list = []

        for min_supp in min_supp_values:
            for min_conf in min_conf_values:
                frequent_itemsets = apriori_algorithm(transactions, min_supp)
                rules = generate_association_rules(frequent_itemsets, min_conf, transactions)
                results_list.append((min_supp, min_conf, rules))
        table_data = []
        for result in results_list:
            min_supp, min_conf, rules = result
            rules_list = [list(rule) for rule in rules]
            rules_json = json.dumps(rules_list)
            table_data.append({
                    'Min Support': min_supp,
                    'Min Confidence': min_conf,
                    'Association Rules': rules_json,
                })
     
    else:
        image_section = html.Div()
        discretized_table = []
        table_data =[]

    

    return table, summary, boxplot, histogram, correlation_matrix, cleaned_table_data, cleaned_boxplot, cleaned_histogram,normalized_table_data,image_section,discretized_table,table_data

if __name__ == '__main__':
    app.run(debug=True)