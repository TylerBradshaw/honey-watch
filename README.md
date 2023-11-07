# honey-watch
Cyber Attack Prediction Dashboard
Overview
This repository contains the code for a web-based dashboard application designed to analyze and predict cyber attack trends using honeypot data. It utilizes historical attack data to provide insights into patterns and predict future activity through a series of interactive visualizations and predictive models.

Features
Data Visualization: Interactive graphs and charts that display the distribution of cyber attacks over time, categorized by type and country.
Time Series Analysis: Utilizes Ordinary Least Squares (OLS) regression to analyze the trend of attacks and to visualize residuals and model diagnostics.
Predictive Modeling: An OLS regression model predicts the number of attacks for future months and years based on historical data.
Customizable Time Range: Users can select specific time ranges to train the model and analyze the trends within those intervals.
Interactive Predictions: Allows users to input a specific month and year to receive predictions on cyber attack occurrences.
Styling and Themes: The dashboard includes styling options, such as color themes, border stylings, and layout adjustments, providing a more personalized user experience.
Technologies
Python: The core language used for data processing and predictive modeling.
SQLite: Database management system used for storing and retrieving historical attack data.
Dash: A Python web framework for building analytical web applications.
Plotly: Graphing library for creating interactive plots.
Statsmodels: Python module that provides classes and functions for the estimation of many different statistical models.
Pandas: Data manipulation and analysis library.
How to Use
Clone the repository.
Install the required dependencies via pip install -r requirements.txt.
Execute the app.py script to launch the dashboard on your local server.
Explore the visualizations and use the prediction feature to forecast future cyber attacks.
Contributing
Contributions to the dashboard are welcome. 
