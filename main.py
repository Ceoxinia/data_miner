from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import base64
import datetime
import io

import pandas as pd
import os

app = Dash(__name__)

df1 = pd.read_csv('Dataset1.csv')
df2 = pd.read_csv('Dataset2.csv')
df3 = pd.read_csv('Dataset3.csv')
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

def normaliser(datasetcleaned,methode):
  dataset=datasetcleaned.copy()
  return dataset

def apriori(datasetcleaned):
    dataset=datasetcleaned.copy()           
    return dataset

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
            html.Button('Clean Data', id='clean-button', n_clicks=0)
        ]),
        html.Div([
            html.H5('Normalization'),
            html.Button('Normalize data', id='clean-button', n_clicks=0)
        ]),
        html.Div([
            html.H5('Application Apriori sur Dataset3'),
            html.Button('Generate Association Rules', id='clean-button', n_clicks=0)
        ]),
        html.Div([
            html.H5('Classification And Clustering'),
            html.H5('Select Classification Algorithme'),
            dcc.Dropdown(
                id='missing-dropdown',
                options=[
                    {'label': 'K-NN', 'value': 'mean'},
                    {'label': 'Decision Tree', 'value': 'median'},
                    {'label': 'Random Forest', 'value': 'mode'}
                ],
                value='mean',
                style={'width': '50%'}
            ),
            html.H5('Select Clustering Methode'),
            dcc.Dropdown(
                id='missing-dropdown',
                options=[
                    {'label': 'K-Means', 'value': 'mean'},
                    {'label': 'DBSCAN', 'value': 'median'},
                ],
                value='mean',
                style={'width': '50%'}
            ),
            html.Button('Generate Association Rules', id='clean-button', n_clicks=0)
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
    
    html.Div(id='image-section')
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
               Output('image-section', 'children')],  # Add this Output for image section
              [Input('data-dropdown', 'value'),
               Input('clean-button', 'n_clicks')],
              [State('outlier-dropdown', 'value'),
               State('missing-dropdown', 'value')])
def update_output(selected_data, n_clicks, selected_outliers, selected_missing):
    df = get_selected_dataframe(selected_data)
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

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
    cleaned_table_data = cleaned_df.to_dict('records'),
 
           
    cleaned_boxplot = px.box(cleaned_df, y=cleaned_df.select_dtypes(include=['number']).columns)
    cleaned_boxplot.update_layout(title_text='Cleaned Boxplot of Numeric Columns')

    cleaned_histogram = px.histogram(cleaned_df, x=cleaned_df.select_dtypes(include=['number']).columns, marginal="rug")
    cleaned_histogram.update_layout(title_text='Cleaned Histogram of Numeric Columns')
    if selected_data == 'df2':
            image_section = html.Div([
            html.H5('Interesting Insights Of Dataset 2:'),
            # Add your HTML components or image tags here for showcasing images
            html.Img(src='/dataset2/confirmedtests.png'),
            html.Img(src='/dataset2/covidevolutionovertime.png'),
            html.Img(src='/dataset2/weekly.png'),
            html.Img(src='/dataset2/q3.png'),
            html.Img(src='/dataset2/q5.png'),
            html.Img(src='/dataset2/q6.png')
        ])
    else:
        image_section = html.Div()    
    

    return table, summary, boxplot, histogram, correlation_matrix, cleaned_table_data[0], cleaned_boxplot, cleaned_histogram,image_section

if __name__ == '__main__':
    app.run(debug=True)