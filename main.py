from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from itertools import combinations
from tabulate import tabulate
import json
import base64
import dash_bootstrap_components as dbc

import pandas as pd
import os
import base64


from classclus import execute_kmeans
from classclus import execute_dbscan
from classclus import execute_knn
from classclus import execute_RF

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


def create_scatter_chart(x_axis="N", y_axis="K"):
    return px.scatter(data_frame=df1, x=x_axis, y=y_axis, height=600, template="plotly_dark",)

columns = ["N", "K", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", 'B', 'OM', 'Fertility']
x_axis = dcc.Dropdown(id="x_axis", options=columns, value="N", clearable=False)
y_axis = dcc.Dropdown(id="y_axis", options=columns, value="K", clearable=False)

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

#===========================App Layout======================================#

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

clustering_algorithms = [
    {'label': 'K-Means', 'value': 'kmeans'},
    {'label': 'DBSCAN', 'value': 'dbscan'}
]

k_options = [
    {'label': '2', 'value': 2},
    {'label': '3', 'value': 3},
    {'label': '4', 'value': 4},
    {'label': '5', 'value': 5}
]

iter_options = [
    {'label': '50', 'value': 50},
    {'label': '100', 'value': 100},
    {'label': '150', 'value': 150},
    {'label': '200', 'value': 200}
]

conv_options = [
    {'label': '0.01', 'value': 0.01},
    {'label': '0.001', 'value': 0.001},
    {'label': '0.0001', 'value': 0.0001}
]

clasification_algorithms = [
    {'label': 'K-NN', 'value': 'Knn'},
    {'label': 'Decision-Trees', 'value': 'DecisinT'},
    {'label': 'Random-Forest', 'value': 'RandomF'}
]

k_nn_options = [
    {'label': '2', 'value': 2},
    {'label': '3', 'value': 3},
    {'label': '4', 'value': 4},
    {'label': '5', 'value': 5}
]

k_nn_dictance = [
    {'label': 'euclidean', 'value': 'euclidean'},
    {'label': 'manhattan', 'value': 'manhattan'},
    {'label': 'chebyshev', 'value': 'chebyshev'},
    {'label': 'cosine', 'value': 'cosine'}
]

rf_estim = [
    {'label': '5', 'value': 5},
    {'label': '10', 'value': 10},
    {'label': '15', 'value': 15},
    {'label': '20', 'value': 20}
]

esp_options = [
    {'label': '0.4', 'value': 0.4},
    {'label': '0.5', 'value': 0.5},
    {'label': '0.6', 'value': 0.6}
]

voisin_options = [
    {'label': '2', 'value': 2},
    {'label': '4', 'value': 4},
    {'label': '6', 'value': 6}
]

table_style = {
    'font-family': 'Arial, sans-serif',
    'border-collapse': 'collapse',
    'width': '100%',
    'margin-top': '10px',
    }

# Créer un style pour les cellules de la DataTable
cell_style = {
    'border': '3px solid #dddddd',
    'text-align': 'left',
    'padding': '8px',
    'fontWeight': 'bold',
}
app.layout = html.Div([

    html.Div(
        className="header",
        style={"backgroundColor": "#3c6382"},
        children=[html.H2(
            "Data Mining Project - Master 2 SII",
            style={
                "color": "white",
                "padding": "30px 0 30px 0",
                "textAlign": "center"}
        )],
    ),

    html.Div([
        html.Div([
        #html.H2("Data Mining Project", style={'color': '#193d8b', 'textAlign': 'center'}),
           
        ]),
        
        # Dataset selection dropdown
        html.H5('Select Dataset', style={'color':'#656668','margin-left': '20px'}),
        dcc.Dropdown(
            id='data-dropdown',
            options=[
                {'label': 'Dataset 1', 'value': 'df1'},
                {'label': 'Dataset 2', 'value': 'df2'},
                {'label': 'Dataset 3', 'value': 'df3'}
            ],
            value='df1',  # default selection
            style={
            'width': '100%',
            'color': '#193d8b',  # Couleur du texte
            'backgroundColor': '#87ceeb',  # Couleur de fond
            'borderColor': '#193d8b'  # Couleur de la bordure
        }
        ),
        
        # Cleaning methods dropdowns and button
        html.Div([
            html.H6('Select Outliers Handling Method',style={'color':'#656668','margin-left': '20px'}),
            dcc.Dropdown(
                id='outlier-dropdown',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': 'Mode', 'value': 'mode'}
                ],
                value='mean',
                style={
            'width': '100%',
            'color': '#193d8b',  # Couleur du texte
            'backgroundColor': '#87ceeb',  # Couleur de fond
            'borderColor': '#193d8b'  # Couleur de la bordure
            }
            ),
            html.H6('Select Missing Values Handling Method',style={'color':'#656668','margin-left': '20px'}),
            dcc.Dropdown(
                id='missing-dropdown',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': 'Mode', 'value': 'mode'}
                ],
                value='mean',
                style={
            'width': '100%',
            'color': '#193d8b',  # Couleur du texte
            'backgroundColor': '#87ceeb',  # Couleur de fond
            'borderColor': '#193d8b'  # Couleur de la bordure
            }
            ),
        ]),
        html.Div([
        html.H6('Normalization Method',style={'color':'#656668','margin-left': '20px'}),
        dcc.Dropdown(
            id='normalization-method-dropdown',
            options=[
                {'label': 'Min-Max', 'value': 'min-max'},
                {'label': 'Z-Score', 'value': 'z-score'}
            ],
            value='min-max',  # Default selection
            style={
            'width': '100%',
            'color': '#193d8b',  # Couleur du texte
            'backgroundColor': '#87ceeb',  # Couleur de fond
            'borderColor': '#193d8b'  # Couleur de la bordure
         }
        ),
        ]),
        html.Div([
        html.H6('Discretization Method For Dataset3',style={'color':'#656668','margin-left': '20px'}),
        dcc.Dropdown(
            id='discretization-method-dropdown',
            options=[
                {'label': 'Equal Width', 'value': 'equal_width'},
                {'label': 'Equal Frequency', 'value': 'equal_frequency'}
            ],
            value='equal_width',  # Default selection
            style={
            'width': '100%',
            'color': '#193d8b',  # Couleur du texte
            'backgroundColor': '#87ceeb',  # Couleur de fond
            'borderColor': '#193d8b'  # Couleur de la bordure
         }
        ),
    ]),
    ], style={'width': '40%', 'float': 'left','margin-left': '10px','margin-right': '70px'}),
    

   
    

    html.Div([
        html.Div(id='output-data-upload'),
        html.Div(id='data-summary'),
        
        #dcc.Graph(id='boxplot'),
        #dcc.Graph(id='histogram'),
        #dcc.Graph(id='correlation-matrix'),  # New graph for correlation matrix
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    html.Div(id='graph-data2-section'),
    #html.Div(id='data-summary'),
    html.Div([
    html.Br(),
    "X-Axis", x_axis, 
    "Y-Axis", y_axis,
    
    dcc.Graph(id="scatter",style={'backgroundColor': '#f0f0f0'}),
    ]),
    
    html.Div([
        dcc.Graph(id='boxplot'),
        dcc.Graph(id='histogram'),
        dcc.Graph(id='correlation-matrix'),
    ], style={'display': 'flex', 'flexDirection': 'row'}),
    html.Div([
        html.H5('Cleaned DataFrame',style={'color': '#031b4b', 'textAlign': 'center', 'margin-bottom': '20px'}),
        dash_table.DataTable(id='cleaned-table',
        style_table={'height': '360px','width': '100%','overflowY': 'auto','margin-top': '15px'},
        style_cell={
        'textAlign': 'center',
        'backgroundColor': '#f0f8ff',  # Couleur de fond des cellules
        'color': '#031b4b'  # Couleur du texte des cellules
            },
        style_header={
        'backgroundColor': '#87ceeb',  # Couleur de fond des en-têtes
        'fontWeight': 'bold'  # Mise en gras du texte des en-têtes
        })
    ]),
    
    html.Div([
        dcc.Graph(id='cleaned-boxplot'),
        dcc.Graph(id='cleaned-histogram')
    ]),
    
    html.Div(id='image-section'),
    html.Div([
    html.H5('Normalized DataFrame',style={'color': '#031b4b', 'textAlign': 'center', 'margin-bottom': '20px'}),
    dash_table.DataTable(id='normalized-table',
    style_table={'height': '360px','width': '100%','overflowY': 'auto','margin-top': '15px'},
    style_cell={
    'textAlign': 'center',
    'backgroundColor': '#f0f8ff',  # Couleur de fond des cellules
    'color': '#031b4b'  # Couleur du texte des cellules
        },
        style_header={
    'backgroundColor': '#87ceeb',  # Couleur de fond des en-têtes
    'fontWeight': 'bold'  # Mise en gras du texte des en-têtes
    })    
]),
    html.Div([
    html.H5('Descritisized DataFrame',style={'color': '#031b4b', 'textAlign': 'center', 'margin-bottom': '20px'}),
    dash_table.DataTable(id='discretization-table',
    style_table={'height': '360px','width': '100%','overflowY': 'auto','margin-top': '15px'},
    style_cell={
    'textAlign': 'center',
    'backgroundColor': '#f0f8ff',  # Couleur de fond des cellules
    'color': '#031b4b'  # Couleur du texte des cellules
        },
        style_header={
    'backgroundColor': '#87ceeb',  # Couleur de fond des en-têtes
    'fontWeight': 'bold'  # Mise en gras du texte des en-têtes
    })    
]),
    html.Div([
    html.H5('Association Rules Table',style={'color': '#031b4b', 'textAlign': 'center', 'margin-bottom': '20px'}),
    dash_table.DataTable(
        id='association-rules-table',
        columns=[
            {'name': 'Min Support', 'id': 'Min Support'},
            {'name': 'Min Confidence', 'id': 'Min Confidence'},
            {'name': 'Association Rules', 'id': 'Association Rules'},
        ],
        style_table={'height': '360px','width': '100%','overflowY': 'auto','margin-top': '15px'},
        style_cell={
        'textAlign': 'left',
        'backgroundColor': '#f0f8ff',  # Couleur de fond des cellules
        'color': '#031b4b'  # Couleur du texte des cellules
            },
            style_header={
        'backgroundColor': '#87ceeb',  # Couleur de fond des en-têtes
        'fontWeight': 'bold'  # Mise en gras du texte des en-têtes
        }
    ),
]),

html.Hr(style={'color': '#193d8b', 'background-color': '#193d8b', 'height': '2px'}),

    # Classification algorithm selection dropdown
    
    html.Div([
        html.H5('Select Classification Algorithm', className='dropdown-label',style={'color':'#656668','margin-left': '20px'}),
        dcc.Dropdown(
            id='classification-algorithm-dropdown',
            options=clasification_algorithms,
            value='',  # default selection
            #style={'width': '35%'},
            style={
            'width': '100%',
            'color': '#193d8b',  # Couleur du texte
            'backgroundColor': '#87ceeb',  # Couleur de fond
            'borderColor': '#193d8b',  # Couleur de la bordure
            'box-shadow': '0px 0px 5px 1px #193d8b'
            }
        ),
    ], className='dropdown-container',style={'width': '40%'}),

    html.Div([
    html.H5('Select Your Instance', className='dropdown-label', style={'color': '#656668', 'margin-left': '20px'}),

    dbc.Row([
        dbc.Col(dcc.Input(id=f'{i}', type='text', placeholder=f'Extra Input {i}', style={'width': '40%'})) for i in range(1, 13)
    ], className='input-row', style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between', 'margin': '20px'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'margin': '20px'}),

    html.Div(id='output-classification',
    style={
            'color': '#193d8b',
            'textAlign': 'center',  # Center-align the text
            'fontSize': '24px',    # Increase the font size
            'margin-top': '20px'   # Add top margin
        } ),

    html.Div([
        html.H5('K-nn Parameters', className='parameter-label', style={'color': '#031b4b', 'margin-top': '10px', 'font-style': 'italic'}),
        dcc.Dropdown(
            id='k-nn-input',
            options=k_nn_options,
            placeholder='Enter k',
            style={'width': '40%'},
        ),
        dcc.Dropdown(
            id='distance-metric',
            options=k_nn_dictance,
            placeholder='Enter distance',
            style={'width': '40%'}
        ),
    ], id='knn-parameters', style={'display': 'block','margin-top': '20px', 'padding': '10px', 'border': '1px solid #193d8b', 'border-radius': '5px'}, className='parameter-container'),

    html.Div([
        html.H5('Random Forest Parameters', className='parameter-label',style={'color': '#031b4b', 'margin-top': '10px', 'font-style': 'italic'}),
        dcc.Dropdown(
            id='estim-input',
            options=rf_estim,
            placeholder='Enter estimation',
            style={'width': '40%'}
        ),
    ], id='rf-parameters', style={'display': 'block','margin-top': '20px', 'padding': '10px', 'border': '1px solid #193d8b', 'border-radius': '5px'}, className='parameter-container'),

    html.Hr(style={'color': '#193d8b', 'background-color': '#193d8b', 'height': '2px'}),

    # Clustering algorithm selection dropdown
    html.Div([
        html.H5('Select Clustering Algorithm', className='dropdown-label',style={'color':'#656668','margin-left': '20px'}),
        dcc.Dropdown(
            id='clustering-algorithm-dropdown',
            options=clustering_algorithms,
            value='',  # default selection
            #style={'width': '35%'},
            style={
            'width': '100%',
            'color': '#193d8b',  # Couleur du texte
            'backgroundColor': '#87ceeb',  # Couleur de fond
            'borderColor': '#193d8b',  # Couleur de la bordure
            'box-shadow': '0px 0px 5px 1px #193d8b'
            }
        ),
    ], className='dropdown-container',style={'width': '40%'}),

    html.Div([
        html.H5('K-Means Parameters', className='parameter-label',style={'color': '#031b4b', 'margin-top': '10px', 'font-style': 'italic'}),
        dcc.Dropdown(
            id='k-input',
            options=k_options,
            placeholder='Enter k',
            style={'width': '40%'}
        ),
        dcc.Dropdown(
            id='n-iterations-input',
            options=iter_options,
            placeholder='Enter iteration',
            style={'width': '40%'}
        ),
        dcc.Dropdown(
            id='convergence-input',
            options=conv_options,
            placeholder='Enter convergence',
            style={'width': '40%'}
        ),
    ], id='kmeans-parameters', style={'display': 'block','margin-top': '20px', 'padding': '10px', 'border': '1px solid #193d8b', 'border-radius': '5px'}, className='parameter-container'),

    html.Div([
        html.H5('DBSCAN Parameters', className='parameter-label',style={'color': '#031b4b', 'margin-top': '10px', 'font-style': 'italic'}),
        dcc.Dropdown(
            id='eps-input',
            options=esp_options,
            placeholder='Enter eps',
            style={'width': '40%'}
        ),
        dcc.Dropdown(
            id='min-samples-input',
            options=voisin_options,
            placeholder='Enter min samples',
            style={'width': '40%'}
        ),
    ], id='dbscan-parameters', style={'display': 'block','margin-top': '20px', 'padding': '10px', 'border': '1px solid #193d8b', 'border-radius': '5px'}, className='parameter-container'),


    #html.Button('Reset', id='reset-button', n_clicks=0),

    html.H5('Resulting Image from Clustering',style={'color':'#656668','margin-left': '20px'}),
   
    html.Div(id='kmeans-result-image'),
    
    
    

    html.Div(
        className="header",
        style={"backgroundColor": "#3c6382"},
        children=[html.H2(
            "All rights reserved - FERKOUS & KHEMISSI - 2024",
            style={
                "color": "white",
                "padding": "30px 0 30px 0",
                "textAlign": "center"}
        )],
    ),
])


def get_selected_dataframe(selected_data):
    if selected_data == 'df1':
        return df1
    elif selected_data == 'df2':
        return df2
    elif selected_data == 'df3':
        return df3

@app.callback(
    [   Output("scatter", "figure"),
        Output('output-data-upload', 'children'),
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
        Output('association-rules-table', 'data'),
        Output('kmeans-result-image', 'children'), #output-classification
        Output('output-classification', 'children'),
        Output('graph-data2-section', 'children'),
    ],
    [   Input("x_axis", "value"),Input("y_axis", "value"),
        Input('data-dropdown', 'value'),
        Input('discretization-method-dropdown', 'value'),
        Input('clustering-algorithm-dropdown', 'value'), #classification-algorithm-dropdown
        Input('classification-algorithm-dropdown', 'value'),
        [Input(f'{i}', 'value') for i in range(1, 13)],
        Input('k-input', 'value'),
        Input('n-iterations-input', 'value'),
        Input('convergence-input', 'value'),
        Input('eps-input', 'value'),
        Input('min-samples-input', 'value'),
        State('outlier-dropdown', 'value'),
        State('missing-dropdown', 'value'),
        State('normalization-method-dropdown', 'value'),
        Input('k-nn-input', 'value'),
        Input('distance-metric', 'value'),
        Input('estim-input', 'value'),
    ]
)
def update_output(
    x,y,selected_data, selected_discretization_method, selected_cluster, selected_clasification, arg,
    selected_k, selected_iter, selected_converg, selected_eps, selected_minSample,
      # Ajout de l'argument pour le bouton de réinitialisation
    selected_outliers, selected_missing, selected_normalization_method, selected_k_knn, selected_distance, selected_estimation,
):
    df = get_selected_dataframe(selected_data)
    # ... rest of the function
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    image_section = html.Div()

    if selected_data=='df1':
        d='Propriétés du Sol'
    elif selected_data == 'df2':
        d='COVID-19'
    else:
        d='Agriculture'        
    table = html.Div([
        html.H5(f'Selected Dataset: {d}',style={'color':'#031b4b','textAlign': 'center','margin-top': '10px'}),
        dash_table.DataTable(
            df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'height': '360px','width': '100%','overflowY': 'auto','margin-top': '15px'},
            style_cell={
            'textAlign': 'center',
            'backgroundColor': '#f0f8ff',  # Couleur de fond des cellules
            'color': '#031b4b'  # Couleur du texte des cellules
             },
             style_header={
            'backgroundColor': '#87ceeb',  # Couleur de fond des en-têtes
            'fontWeight': 'bold'  # Mise en gras du texte des en-têtes
             }
        ),
    ])

    

    conditional_style = [
    {
        'if': {'row_index': 'odd'},
        'backgroundColor': '#e6f7ff',  # Bleu clair
    }
    ]
    summary = html.Div([
    html.Hr(style={'color': '#193d8b', 'background-color': '#193d8b', 'height': '2px'}),
    
    html.H5('DataFrame Summary', style={'color': '#031b4b', 'textAlign': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.H6('Description des colonnes:', style={'color': '#031b4b', 'textAlign': 'center', 'margin-bottom': '20px'}),
            dash_table.DataTable(
                data=get_column_description(df).to_dict('records'),
                columns=[{'name': col, 'id': col} for col in get_column_description(df).columns],
                style_table=table_style,
                style_cell=cell_style,
                style_data_conditional=conditional_style,
            ),
        ], style={'width': '60%', 'padding': '20px', 'margin-right': '200px'}),  # Ajout de margin-right
        html.Div([
            html.H6('Les Tendances Centrales:', style={'color': '#031b4b', 'textAlign': 'center', 'margin-bottom': '20px'}),
            dash_table.DataTable(
                data=df.describe().transpose().reset_index().to_dict('records'),
                columns=[{'name': col, 'id': col} for col in df.describe().transpose().reset_index().columns],
                style_table=table_style,
                style_cell=cell_style,
                style_data_conditional=conditional_style,
            ),
        ], style={'width': '60%', 'padding': '20px'})
    ], style={'display': 'flex', 'justify-content': 'flex-end'})
])

    boxplot = px.box(df, y=df.select_dtypes(include=['number']).columns)
    boxplot.update_layout(
    title_text='Boxplot of Numeric Columns',
    xaxis_title='X-Axis Title',
    yaxis_title='Y-Axis Title',
    font=dict(family="Arial, sans-serif", size=12, color="rgb(255, 255, 0)"),
    paper_bgcolor='black',
    plot_bgcolor='black'
    )

    histogram = px.histogram(df, x=df.select_dtypes(include=['number']).columns, marginal="rug")
    histogram.update_layout(
    title_text='Histogram of Numeric Columns',
    xaxis_title='X-Axis Title',
    yaxis_title='Y-Axis Title',
    font=dict(family="Arial, sans-serif", size=12, color="rgb(255, 255, 0)"),
    paper_bgcolor='black',
    plot_bgcolor='black'
    )


    # Correlation matrix graph
    correlation_matrix = px.imshow(df_numeric.corr(), labels=dict(x='Columns', y='Columns'), x=df.columns, y=df.columns)
    correlation_matrix.update_layout(
    title_text='Correlation Matrix',
    xaxis_title='X-Axis Title',
    yaxis_title='Y-Axis Title',
    font=dict(family="Arial, sans-serif", size=12, color="rgb(255, 255, 0)"),
    paper_bgcolor='black',
    plot_bgcolor='black'
   )


    cleaned_df = clean_dataset(df, selected_missing, selected_outliers)
    cleaned_table_data = cleaned_df.to_dict('records')
 
           
    cleaned_boxplot = px.box(cleaned_df, y=cleaned_df.select_dtypes(include=['number']).columns)
    #cleaned_boxplot.update_layout(title_text='Cleaned Boxplot of Numeric Columns')
    cleaned_boxplot.update_layout(
    title_text='Cleaned Boxplot of Numeric Columns',
    xaxis_title='X-Axis Title',
    yaxis_title='Y-Axis Title',
    font=dict(family="Arial, sans-serif", size=12, color="rgb(255, 255, 0)"),
    paper_bgcolor='black',
    plot_bgcolor='black'
    )

    cleaned_histogram = px.histogram(cleaned_df, x=cleaned_df.select_dtypes(include=['number']).columns, marginal="rug")
    #cleaned_histogram.update_layout(title_text='Cleaned Histogram of Numeric Columns')
    cleaned_histogram.update_layout(
    title_text='Cleaned Histogram of Numeric Columns',
    xaxis_title='X-Axis Title',
    yaxis_title='Y-Axis Title',
    font=dict(family="Arial, sans-serif", size=12, color="rgb(255, 255, 0)"),
    paper_bgcolor='black',
    plot_bgcolor='black'
)

    
    normalized_df = normaliser(cleaned_df, selected_normalization_method)
    normalized_table_data = normalized_df.to_dict('records')
    image_section3=''
    if selected_data == 'df2':
        image_section3 = html.Div([
        html.H5('Interesting Insights Of Dataset 2:',style={'color': '#031b4b', 'textAlign': 'center', 'margin-bottom': '20px'}),
        html.Img(src=b64_image('assets/confirmedtests.png'),style={'width': '50%', 'height': 'auto'}),
        html.Img(src=b64_image('assets/covidevolutionovertime.png'),style={'width': '50%', 'height': 'auto'}),
        html.Img(src=b64_image('assets/dispersion.png'),style={'width': '50%', 'height': 'auto'}),
        html.Img(src=b64_image('assets/weekly.png'),style={'width': '50%', 'height': 'auto'}),
        html.Img(src=b64_image('assets/q3.png'),style={'width': '50%', 'height': 'auto'}),
        html.Img(src=b64_image('assets/q5.png'),style={'width': '50%', 'height': 'auto'}),
        html.Img(src=b64_image('assets/q6.png'),style={'width': '50%', 'height': 'auto'}),   
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
    image_path = ''
    if selected_cluster == 'kmeans':
        if selected_k is not None and selected_iter is not None and selected_converg is not None:
            # Convertir les valeurs en entiers
            selected_k = int(selected_k)
            selected_iter = int(selected_iter)
            selected_converg = float(selected_converg)
            
            print('avant')
            # Appeler la fonction execute_kmeans avec les paramètres fournis
            image_path = execute_kmeans(selected_k, selected_iter, selected_converg)
            print('apres')
    if selected_cluster == 'dbscan':
        if selected_eps is not None and selected_minSample is not None:
            # Convertir les valeurs en entiers
            selected_eps = float(selected_eps)
            selected_minSample = int(selected_minSample)
            
            print('avant 2')
            # Appeler la fonction execute_dbscan avec les paramètres fournis
            print(selected_eps)
            print(selected_minSample)
            image_path = execute_dbscan(selected_eps, selected_minSample)
            print('apres 2')
    # Convert the image to base64
    image_section2=''
    if image_path:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        # Set the image as a child of the `image-section`
        image_section2 = html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '50%'})
    else:
        print('vide')
        #image_section = html.Div()
    resultat_class=''
    if selected_clasification == 'Knn': 
        #print('ici: ',arg)
        if None not in arg:
            if selected_k_knn is not None and selected_distance is not None:
                print('ici: ',arg)
                r = execute_knn(arg, selected_k_knn, selected_distance)
                resultat_class = 'The class associated with this instance is: '+str(r)
                print(resultat_class)

    
    if selected_clasification == 'RandomF':
        print('ici: ',arg)
        if None not in arg:
            if selected_estimation is not None:
                print('ici: ',arg)
                f = execute_RF(arg, selected_estimation)
                resultat_class = 'The class associated with this instance is: '+str(f)
                print(resultat_class)
                
            
    return create_scatter_chart(x, y),table, summary, boxplot, histogram, correlation_matrix, cleaned_table_data, cleaned_boxplot, cleaned_histogram, normalized_table_data, image_section, discretized_table, table_data, image_section2, resultat_class, image_section3




if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, threaded=False)
