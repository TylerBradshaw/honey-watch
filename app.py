import sqlite3 as sql
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc, State
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import statsmodels.api as sm
import numpy as np
import calendar

# Connect to the database and fetch data
database = "honeypot_data.db"
connection = sql.connect(database)

data_dict = {}
for year in range(2013, 2023):
    query = f"SELECT *, ((year - 2013) * 12 + month - 1) AS Time_Index FROM honeypot_data_{year}"
    data_dict[year] = pd.read_sql(query, connection)

connection.close()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR], suppress_callback_exceptions=True)
server = app.server

global MODEL_COEFFICIENTS
# Model Coefficients based on 2013-2021 test data. Can be dynamically updated by selecting a different data set.
MODEL_COEFFICIENTS = {
    'const': 418.7474,
    'Time_Index': 1.9485
}

# Sidebar Layout
sidebar_content = html.Div(
    [
        html.Img(src=app.get_asset_url('HoneyWatch_Text.png'), style={'width': '100%', 'height': 'auto'}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Main Dashboard", href="/", active="exact"),
                dbc.NavLink("Regression Analysis", href="/page-1", active="exact"),
                dbc.NavLink("Future Predictions", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
            className="flex-column",
        ),
        html.Hr(),
        html.Div(id='details', className="text-white"),
        html.Div(html.Img(src=app.get_asset_url('HoneyWatch_Logo.png'), style={'width': '100%', 'margin-top': 'auto'})),
    ],
    style={
        'position': 'relative',
        'padding': '10px 20px',
        'border-right': '2px solid #fff',
        'background-color': '#000019',
        'height': '100vh',
        'width': '300px',
        'display': 'flex',
        'flex-direction': 'column'
    }
)

# Define the main content
content = html.Div(id="page-content", style={'padding': '1rem'})

app.layout = dbc.Container(
    [
        dcc.Location(id='url', refresh=False),
        dbc.Row(
            [
                dbc.Col(sidebar_content, md=2, style={'height': '100vh', 'overflowY': 'auto'}),
                dbc.Col(content, md=10),
            ],
            className="g-0",
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    all_years = list(range(2013, 2023))
    if pathname == "/":
        return [
            dbc.Row(
                dbc.Col(
                    html.Div([
                        html.Label("Select Years for Monthly Distribution:"),
                        dcc.Dropdown(
                            id='year-dropdown-monthly',
                            options=[{'label': str(year), 'value': year} for year in all_years],
                            value=all_years,
                            multi=True
                        ),
                        dcc.Graph(id='graph-output-monthly'),
                    ], className="border p-3"),
                    width=12,
                ),
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([
                            html.Label("Select Years for Attack Types:"),
                            dcc.Dropdown(
                                id='year-dropdown-attacktype',
                                options=[{'label': str(year), 'value': year} for year in all_years],
                                value=[2013],
                                multi=True
                            ),
                            dcc.Graph(id='graph-output-attacktype'),
                        ], className="border p-3"),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div([
                            html.Label("Select Years for Attacks by Country:"),
                            dcc.Dropdown(
                                id='year-dropdown-country',
                                options=[{'label': str(year), 'value': year} for year in all_years],
                                value=[2013],
                                multi=True
                            ),
                            dcc.Graph(id='graph-output-country'),
                        ], className="border p-3"),
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        id='map-container',
                        children=[
                            html.Label("Mapbox Map - Coming Soon"),
                            # Placeholder for future map. Abandoned for the time being due to performance issues.
                        ],
                        className="border p-3"
                    ),
                    width=12,
                ),
                className="mb-3",
            ),
        ]
    elif pathname == "/page-1":
        return page_1_layout
    elif pathname == "/page-2":
        return page_2_layout
    # If the user tries to reach a different page, return a 404 message
    if pathname not in ["/", "/page-1", "/page-2"]:
        return dbc.Container(
            dbc.Row(
                dbc.Col(
                    [
                        html.H1("404: Not found", className="text-danger"),
                        html.Hr(),
                        html.H4(f"The pathname {pathname} was not recognized..."),
                    ],
                    className="py-5 my-5 text-center",
                    width={"size": 8, "offset": 2},
                ),
            ),
            className="h-100 p-5 bg-light border rounded-3",
            fluid=True,
            style={"height": "100vh"},
        )


# Layout for page OLS Regression Analysis
page_1_layout = html.Div([
    html.H2('OLS Regression Analysis', style={'textAlign': 'center'}),
    html.Div(
        dcc.Dropdown(
            id='time-range-dropdown',
            options=[
                {'label': '2013-2015', 'value': '2013-2015'},
                {'label': '2016-2018', 'value': '2016-2018'},
                {'label': '2019-2021', 'value': '2019-2021'},
                {'label': '2013-2017', 'value': '2013-2017'},
                {'label': '2018-2021', 'value': '2018-2021'},
                {'label': '2013-2021', 'value': '2013-2021'},
            ],
            value='2013-2021',  # Default value
            style={'width': '50%', 'margin': '10px auto'}
        ), style={'textAlign': 'center'}
    ),
    dcc.Graph(
        id='ols-diagnostic-plot',
        style={'height': '500px'},
        config={
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': 'OLS_diagnostic_plot'
            }
        }
    ),
    html.Div(
        id='ols-summary-statistics',
        style={'margin': '20px auto', 'width': '80%', 'textAlign': 'center'}
    ),
    html.H2('Test OLS Model with 2022 Data', style={'textAlign': 'center'}),
    html.Div(
        html.Button('Test Model', id='test-model-button', n_clicks=0),
        style={'textAlign': 'center', 'padding': '20px'}
    ),
    dcc.Graph(
        id='test-results-plot',
        style={'height': '500px'},
        config={
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': 'test_results_plot'
            }
        }
    ),
    html.Div(id='test-metrics-output', style={'textAlign': 'center', 'margin': '20px'}),
], style={'margin': 'auto', 'width': '80%', 'padding': '10px'})

# Layout for Future Predictions
page_2_layout = html.Div([
    html.Div(
        [
            html.H2('Future Attack Predictions', style={'textAlign': 'center'}),
            dcc.Input(
                id='input-month',
                type='number',
                placeholder='Month (1-12)',
                style={'marginRight': '10px'}
            ),
            dcc.Input(
                id='input-year',
                type='number',
                placeholder='Year',
                style={'marginRight': '10px'}
            ),
            html.Button('Predict', id='submit-button', n_clicks=0)
        ],
        style={
            'textAlign': 'center',
            'padding': '20px',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'margin': '10px',
            'backgroundColor': 'rgb(0,0,0,0)'
        }
    ),
    html.Div(
        id='prediction-output',
        style={
            'textAlign': 'center',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'margin': '10px',
            'padding': '20px',
            'backgroundColor': 'rgb(0,0,0,0)'
        }
    )
],
style={'textAlign': 'center'})



# Function to predict future traffic patterns based on month and year
def predict_traffic_direct(month, year):
    # Calculate the time index for the given month and year
    base_date = pd.Timestamp('2013-01-01')
    target_date = pd.Timestamp(year=year, month=month, day=1)
    # Calculate the total number of months difference
    time_index = ((target_date.year - base_date.year) * 12 + target_date.month - base_date.month)

    # Calculate the prediction using the model coefficients
    predicted_attacks = MODEL_COEFFICIENTS['const'] + MODEL_COEFFICIENTS['Time_Index'] * time_index
    return predicted_attacks


def time_index_to_month_year(time_index, start_year=2013):
    # Calculate the year and month from the time index
    year = start_year + time_index // 12
    month = time_index % 12 + 1  # +1 because % 12 returns 0 for December
    month_name = calendar.month_abbr[month]
    return f"{month_name}-{year}"


@app.callback(
    [Output('test-results-plot', 'figure'),
     Output('test-metrics-output', 'children')],
    [Input('test-model-button', 'n_clicks'),
     Input('time-range-dropdown', 'value')],
)
def test_model_callback(n_clicks, time_range):
    if n_clicks > 0:  # Ensure the function executes only when the button is clicked
        return test_model(n_clicks, time_range)
    return go.Figure(), html.Div()  # Return empty figure and div if button not clicked


def test_model(n_clicks, time_range):
    start_year, end_year = map(int, time_range.split('-'))

    with sql.connect(database) as conn:
        # Prepare the SQL queries based on the selected time range
        train_data_frames = []
        for year in range(start_year, end_year + 1):
            query = f"""
                SELECT *, ((year - {start_year}) * 12 + month - 1) AS Time_Index 
                FROM honeypot_data_{year}
                """
            df = pd.read_sql(query, conn)
            train_data_frames.append(df)

        # Concatenate and calculate the number of attacks
        training_data = pd.concat(train_data_frames, ignore_index=True)
        training_data['Num_Attacks'] = training_data.groupby('Time_Index')['Time_Index'].transform('count')

        # Prepare test data for 2022 with Time_Index
        test_data = pd.read_sql("SELECT *, ((year - 2013) * 12 + month - 1) AS Time_Index FROM honeypot_data_2022",
                                conn)
        test_data['Num_Attacks'] = 1  # Each row is an attack
        test_data = test_data.groupby('Time_Index').sum().reset_index()

    # Fit the OLS model on training data
    X_train = sm.add_constant(training_data['Time_Index'])
    y_train = training_data['Num_Attacks']
    model = sm.OLS(y_train, X_train).fit()

    # Prepare the test data
    X_test = sm.add_constant(test_data['Time_Index'])
    y_test = test_data['Num_Attacks']
    predictions = model.predict(X_test)

    # Check if any NaN values in predictions or y_test
    if predictions.isna().any() or y_test.isna().any():
        return go.Figure(), html.Div([html.P("Prediction resulted in NaN values.")])

    # Calculate metrics, handling potential NaN values
    mse = ((predictions - y_test) ** 2).mean(skipna=True)
    rmse = np.sqrt(mse)
    mae = (predictions - y_test).abs().mean(skipna=True)

    # Ensure we have no NaN values in metrics
    if np.isnan(mse) or np.isnan(rmse) or np.isnan(mae):
        return go.Figure(), html.Div([
            html.P("Error calculating metrics: NaN values encountered.")
        ])

    # Convert Time_Index to month-year format for plotting
    test_data['Month_Year'] = test_data['Time_Index'].apply(lambda x: time_index_to_month_year(x, start_year=2013))

    # Create a figure to show actual vs. predicted values for 2022
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Month_Year'], y=y_test, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=test_data['Month_Year'], y=predictions, mode='lines', name='Predicted'))
    fig.update_layout(
        title='Actual vs Predicted Attacks for 2022',
        xaxis_title='Month-Year',
        yaxis_title='Num Attacks'
    )

    # Format for display
    metrics_output = html.Div([
        html.P(f"MSE: {mse:.2f}"),
        html.P(f"RMSE: {rmse:.2f}"),
        html.P(f"MAE: {mae:.2f}")
    ])

    return fig, metrics_output


@app.callback(
    [Output('ols-diagnostic-plot', 'figure'),
     Output('ols-summary-statistics', 'children')],
    [Input('time-range-dropdown', 'value')]
)
def update_ols_analysis(time_range):
    global MODEL_COEFFICIENTS
    # Extract start and end years from the selected time range
    start_year, end_year = map(int, time_range.split('-'))

    # Aggregate data for the selected time range from the data_dict
    filtered_data = pd.concat([data_dict[year] for year in range(start_year, end_year + 1) if year in data_dict])

    # We count the rows for each Time_Index
    attack_counts = filtered_data.groupby('Time_Index').size().reset_index(name='Num_Attacks')

    # Reset the index on both X and y to ensure they are aligned
    X = sm.add_constant(attack_counts['Time_Index']).reset_index(drop=True)
    y = attack_counts['Num_Attacks'].reset_index(drop=True)

    # Now we perform the OLS regression with this count
    ols_model = sm.OLS(y, X).fit()
    MODEL_COEFFICIENTS['const'] = ols_model.params['const']
    MODEL_COEFFICIENTS['Time_Index'] = ols_model.params['Time_Index']

    # Update labels
    month_year_labels = attack_counts['Time_Index'].apply(time_index_to_month_year)

    # Create the OLS diagnostic plot
    residual_plot = go.Figure()
    residual_plot.add_trace(go.Scatter(
        x=month_year_labels,
        y=ols_model.resid,
        mode='markers'
    ))
    residual_plot.update_layout(
        title='Residuals Plot',
        xaxis_title='Month-Year',
        yaxis_title='Residuals',
        xaxis=dict(tickvals=attack_counts['Time_Index'], ticktext=month_year_labels)
    )

    # Generate the OLS summary statistics (convert the summary to HTML)
    ols_summary_html = html.Pre(str(ols_model.summary()))

    return residual_plot, ols_summary_html


def compute_statistics_db(years, country):
    # Reconnect to the database
    with sql.connect(database) as conn:
        # Fetch the data
        dataframes = []
        for year in years:
            # Ensure year is an integer to prevent SQL injection
            year = int(year)
            table_name = f"honeypot_data_{year}"
            query = f"SELECT * FROM {table_name} WHERE country = ?"
            df_year = pd.read_sql(query, conn, params=(country,))
            dataframes.append(df_year)

        # Combine all dataframes into one
        if dataframes:
            df_country = pd.concat(dataframes, ignore_index=True)
        else:
            df_country = pd.DataFrame()

    # Count the attack types
    attack_types_counts = df_country['attacktype'].value_counts().to_dict()

    # Finding the most attacked honeypot
    most_attacked_honeypot = df_country['host'].value_counts().idxmax()
    most_attacked_honeypot_count = df_country['host'].value_counts().max()

    # Get the top 3 IP addresses in terms of attack frequency
    top_ips = df_country['src'].value_counts().nlargest(3).to_dict()

    return attack_types_counts, most_attacked_honeypot, most_attacked_honeypot_count, top_ips


@app.callback(
    Output('details', 'children'),
    [Input('graph-output-country', 'hoverData')],
    [State('year-dropdown-country', 'value')]
)
def display_attack_details(hoverData, selected_years):
    if hoverData:
        # Extract the country name from the hoverData
        country_name = hoverData['points'][0]['label']

        # Handle case where no year is selected
        if not selected_years:
            # If year has not been selected, return a message asking to select a year
            return [html.P("Please select a year to view attack details.")]

        # If year has been selected, proceed with fetching the statistics
        attack_types_counts, most_attacked_honeypot, most_attacked_honeypot_count, top_ips = compute_statistics_db(
            selected_years, country_name)

        # Format the results to display
        details_components = [html.H5(f"Attack Summary for {country_name}:")]
        details_components += [html.P(f"{attack}: {count}") for attack, count in attack_types_counts.items()]
        details_components.append(html.Hr())
        details_components.append(
            html.P(f"Most Attacked Honeypot: {most_attacked_honeypot} (Count: {most_attacked_honeypot_count})"))
        details_components += [html.H6(f"Top Threats:")]
        details_components += [html.P(f"{ip}: {count} attacks") for ip, count in top_ips.items()]

        # Return components
        return details_components

    # Default message
    return [html.P("Hover over a country to view attack details.")]


@app.callback(
    Output('prediction-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-month', 'value'), State('input-year', 'value')]
)
def update_prediction(n_clicks, input_month, input_year):
    # Validation check for the year input
    if n_clicks is not None and input_year is not None:
        try:
            # If the month is None, predict for the whole year
            if input_month is None:
                predictions = []
                for month in range(1, 13):  # Loop over all months
                    prediction = predict_traffic_direct(month, int(input_year))
                    predictions.append(f"{month}/{input_year}: {prediction:.2f} attacks")
                return html.Ul([html.Li(prediction) for prediction in predictions])
            else:
                # Predict for the specific month
                prediction = predict_traffic_direct(int(input_month), int(input_year))
                return f'The predicted number of attacks for {input_month}/{input_year} is: {prediction:.2f}'
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "An error occurred while making the prediction."
    return 'Enter a year to get a prediction, and optionally a month for monthly prediction.'


@app.callback(
    [Output('graph-output-attacktype', 'figure'),
     Output('graph-output-monthly', 'figure'),
     Output('graph-output-country', 'figure')],
    [Input('year-dropdown-attacktype', 'value'),
     Input('year-dropdown-monthly', 'value'),
     Input('year-dropdown-country', 'value')]
)
def update_graphs(selected_years_attacktype, selected_years_monthly, selected_years_country):
    # Helper function to aggregate data for selected years
    def aggregate_data(selected_years, data_key):
        if 'All' in selected_years or not selected_years:
            selected_years = data_dict.keys()  # Use all years if All is selected or no year is selected
        return pd.concat([data_dict[year] for year in selected_years if year in data_dict])

    # Attack Types Distribution
    df_attacktype = aggregate_data(selected_years_attacktype, 'attacktype')
    fig_attacktype = px.histogram(df_attacktype, x="attacktype",
                                  title=f"Attack Types Distribution for {', '.join(map(str, selected_years_attacktype))}")

    fig_attacktype.update_layout(
        plot_bgcolor='rgb(0,0,0,0)',
        paper_bgcolor='rgb(0,0,0,0)',
        font_color='white',
        xaxis=dict(color='white'),
        yaxis=dict(color='white')
    )

    # Monthly Attack Distribution
    df_monthly = aggregate_data(selected_years_monthly, 'monthly')
    fig_monthly = px.line(title=f"Monthly Attacks Distribution for {', '.join(map(str, selected_years_monthly))}")
    for year in selected_years_monthly:
        monthly_counts = df_monthly[df_monthly['year'] == year].groupby('month').size()
        fig_monthly.add_scatter(x=monthly_counts.index, y=monthly_counts.values, mode='lines', name=str(year))
    fig_monthly.update_layout(
        plot_bgcolor='rgb(0,0,0,0)',
        paper_bgcolor='rgb(0,0,0,0)',
        font_color='white',
        xaxis=dict(color='white'),
        yaxis=dict(color='white')
    )

    # Attack by Country
    df_country = aggregate_data(selected_years_country, 'country')
    country_counts = df_country.groupby('country').size().sort_values(ascending=False)
    fig_country = px.pie(names=country_counts.index, values=country_counts.values,
                         title=f"Attacks by Country for {', '.join(map(str, selected_years_country))}")
    fig_country.update_layout(
        plot_bgcolor='rgb(0,0,0,0)',
        paper_bgcolor='rgb(0,0,0,0)',
        font_color='white'
    )

    return fig_attacktype, fig_monthly, fig_country


if __name__ == '__main__':
    app.run_server(debug=True)
