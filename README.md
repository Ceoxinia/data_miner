# Data Mining Project - Master 2 SII
## Overview
This Dash project provides a user interface for data exploration, preprocessing, and classification/clustering using various algorithms. The application is designed to handle three datasets (Dataset1, Dataset2, and Dataset3) and offers functionalities such as outlier handling, missing value imputation, normalization, discretization, and classification/clustering.

## Getting Started
Install the required Python packages using 
pip install -r requirements.txt.

Run the Dash app using python main.py.


Access the app in your browser at http://127.0.0.1:8050/.

## Project Structure

main.py: The main script containing the Dash application.

classclus.py: Module containing functions for clustering and classification algorithms.

Dataset1.csv, Dataset2.csv, Dataset3.xlsx: Sample datasets for testing the application.
## Dependencies
Dash: Web framework for building analytical web applications.


Plotly: Graphing library for creating interactive plots.


Seaborn, Matplotlib: Data visualization libraries.


Pandas, Numpy: Data manipulation and analysis.

Scikit-learn: Machine learning library.

Dash Bootstrap Components: Bootstrap components for Dash apps.

Tabulate: Pretty-print tabular data.
## How to Use
Select a dataset (Dataset1, Dataset2, or Dataset3) from the dropdown.

Choose outlier handling and missing value imputation methods.

Select normalization and discretization methods (if applicable).

Explore the dataset using boxplots, histograms, and a correlation ma
trix.

Perform clustering using K-Means or DBSCAN algorithms with customizable parameters.

Apply classification algorithms (K-NN, Decision Trees, Random Forest) with additional parameters.

## User Interface
Header: Displays the project title.

Dataset Selection: Choose the dataset from the dropdown.

Cleaning Methods: Select outlier handling and missing value imputation methods.

Normalization Method: Choose between Min-Max and Z-Score normalization.

Discretization Method (For Dataset3): Choose between Equal Width and Equal Frequency discretization.

Graphs and Tables: Visualize boxplots, histograms, and correlation matrices.

View cleaned, normalized, and discretized data.

Classification Section: Choose a classification algorithm, input instance values, and see the predicted class.

Clustering Section: Choose a clustering algorithm, set parameters, and view resulting images.

## Note
For Dataset2, interesting insights are displayed along with visualizations.

For Dataset3, association rules are generated based on discretized data.

## Acknowledgments
This project is part of the Data Mining course in the Master 2 SII program.
All rights reserved - FERKOUS & KHEMISSI - 2024.
