import warnings
warnings.filterwarnings('ignore')

from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from os.path import join
import pandas as pd
import numpy as np
import pathlib

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import dash
import plotly.graph_objects as go

import torch
import plotly.figure_factory as ff


################################### Initialize app ###################################
app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
server = app.server

# load data
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# EDA
bay_data = pd.read_csv(join(APP_PATH, 'bay_data.csv'))
sd_data = pd.read_csv(join(APP_PATH, 'sd_data.csv')) 
traffic_data = pd.read_csv(join(APP_PATH, 'traffic.csv')) 
traffic_covid_data = pd.read_csv(join(APP_PATH, 'traffic_covid.csv')) 

# traffic predictions
data_pred = pd.read_pickle(join(APP_PATH, 'traffic_pred_horizon12.pkl')) 

end_date = max(data_pred['time'])
start_date = end_date - relativedelta(days=1)

stations_list = data_pred[1].keys()

################################### layout functions ###################################

# main dashboard description and EDA
def main_title_buttons():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.Div([
                            html.H1(children="Environmental Radiation Monitoring in the US"),
                            html.Br(),
                            html.P(
                                children="This dashboard has been created to serve as a platform for radiation monitoring data exploration. \
                                All data can be obtained directly from United States Environmental Protection Agency (EPA). \
                                    Web Link: https://www.epa.gov/radnet/radnet-csv-file-downloads"),
                            html.P(
                                children="All of the data presented here is part of the EPA's RadNet program. The RadNet program monitors \
                                    environmental radiation in air, rain, and drinking water. Scientist can and have used this information \
                                        to track variations in background radiation, atmospheric nuclear weapons, and nuclear reactor accidents."),
                            html.P('Information presented in this map is grouped by the selecion on the "Group By" option. The map and charts will \
                                    update accordingly.'),
                            html.Br(),
                        ]),
                        html.Div([
                            html.P('Select Date Range:', style={'font-weight': 'bold'}),
                            dcc.DatePickerRange(
                                id='date-range12345',
                                start_date_placeholder_text="Start Period",
                                end_date_placeholder_text="End Period",
                                calendar_orientation='vertical',
                                min_date_allowed=min(data_pred['time']),
                                max_date_allowed=max(data_pred['time']),
                                start_date=start_date,
                                end_date=end_date,
                                day_size=45,
                            )
                        ], style={'width': '21%', 'margin-left': '2%', 'color': 'white'}),
                        html.Div([
                            html.P('Group By:', style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id="group-by",
                                options=[
                                    {'label': 'Year', 'value': 'year'},
                                    {'label': 'Month', 'value': 'month'},
                                    {'label': 'Day', 'value': 'day'}
                                ],
                                multi=False,
                                clearable=False,
                                value='day'
                            )  
                        ], style={'width': '15%', 'margin-left': '2%', 'color': 'white'}),
                        html.Div([
                            html.P('Select Radiation Metric:', style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id="metric-select",
                                options=[
                                    {'label': 'Dose Rate [nSv/h]', 'value': 'Dose_Rate'},
                                    {'label': 'Gamma Count Rate [cpm]', 'value': 'Gamma_Count'}
                                ],
                                multi=False,
                                clearable=False,
                                value='Dose_Rate'
                            )  
                        ], style={'width': '18%', 'margin-left': '2%', 'color': 'white'})
                    ], className="row"),
                ], style={'textAlign': 'left', 'margin-left': '2%', 'margin-right': '2%', 'color': 'white'}) 
            ])
        ),
    ])


# prediction analysis buttons
def timeseries_title_buttons():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                	html.Div([
	                    html.Div([
	                        html.H1(children="Environmental Radiation Monitoring in the US"),
	                        html.Br(),
	                        html.P(
	                            children="This dashboard has been created to serve as a platform for radiation monitoring data exploration. \
	                            All data can be obtained directly from United States Environmental Protection Agency (EPA). \
	                                Web Link: https://www.epa.gov/radnet/radnet-csv-file-downloads"),
	                        html.P(
	                            children="All of the data presented here is part of the EPA's RadNet program. The RadNet program monitors \
	                                environmental radiation in air, rain, and drinking water. Scientist can and have used this information \
	                                    to track variations in background radiation, atmospheric nuclear weapons, and nuclear reactor accidents."),
	                        html.P('Information presented in this map is grouped by the selecion on the "Group By" option. The map and charts will \
	                                update accordingly.'),
	                        html.Br(),
	                    ], style={'width': '100%', 'color': 'white'}),
                        html.Div([
                            html.P('Select Date Range:', style={'font-weight': 'bold', 'color': 'white'}),
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date_placeholder_text="Start Period",
                                end_date_placeholder_text="End Period",
                                calendar_orientation='vertical',
	                            min_date_allowed=min(data_pred['time']),
	                            max_date_allowed=max(data_pred['time']),
                                start_date=start_date,
                                end_date=end_date,
                                day_size=45,
                            )
                        ], style={'width': '21%', 'margin-left': '2%'}),	                    
	                    html.Div([
	                        html.P('Select Horizons (5 mins/pts):', style={'font-weight': 'bold', 'color': 'white'}),
							dcc.Dropdown(
							    id='horizons-multiselect',
							    options=[{'label':i+1, 'value':i+1} for i in range(12)],
							    value=[1, 6, 12],
							    multi=True
							) 
	                    ], style={'width': '20%', 'margin-left': '2%'}),
	                    html.Div([
	                        html.P('Select Station:', style={'font-weight': 'bold', 'color': 'white'}),
						    dcc.Dropdown(
						        id='station-dropdown',
						        options=[{'label':i, 'value':i} for i in stations_list],
						        value='400001'
						    )
	                    ], style={'width': '20%', 'margin-left': '2%', 'margin-right': '2%'}),
	                ], className="row"),
                ], style={'textAlign': 'left', 'margin-left': '2%', 'margin-right': '2%'}) 
            ])
        ),
    ])

# functions for plots

def traffic_animated_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="traffic-animated-plot"
                ) 
            ])
        ),  
    ])

def traffic_covid():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="traffic-covid-plot"
                ) 
            ])
        ),  
    ])


def correlations_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="correlations-plot"
                )
            ]), 
        ),  
    ])


def timeseries_plot1():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="timeseries-plot1"
                ) 
            ])
        ),  
    ])


def station_timeseries():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="station-timeseries-plot"
                ) 
            ])
        ),  
    ])

################################### layout ###################################

# App layout
app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    main_title_buttons()
                ], width=12),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    traffic_animated_plot() 
                ], width=7),
                dbc.Col([
                    correlations_plot()
                ], width=5),
            ], align='center'),  
            html.Br(), 
            dbc.Row([
                dbc.Col([
                    traffic_covid() 
                ], width=12),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    timeseries_title_buttons(),
                ], width=12)
            ], align='center'),  
            html.Br(),
            dbc.Row([
                dbc.Col([
                    timeseries_plot1() 
                ], width=6),
                dbc.Col([
                    station_timeseries()
                ], width=6),
            ], align='center'),   
        ]), color = 'dark'
    ),
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='agg-df', style={'display': 'none'})
])

################################### helper functions ###################################

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()

    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0

    return loss.mean()


################################### callback functions ###################################

@app.callback(
    Output("station-timeseries-plot", "figure"),
    [
        Input("station-dropdown", "value"),
        Input("horizons-multiselect", "value"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ]
)
def plot_station_timeseries(station, horizon_list, start_date, end_date):
        
	colors = ['#EF553B', '#636EFA', '#00CC96', '#AB63FA', '#FFA15A', 
          	  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
              '#2CA02C', '#8C564B', '#7F7F7F']

	fig = go.Figure()
	idx = 0
	for h in horizon_list:

	    df = data_pred[h][station]
	    df['time'] = data_pred['time']
	    df_temp = df[(df.time>=start_date) & (df.time<=end_date)] 
	        
	    if idx==0:
	        y_truth = df_temp.truth
	        fig.add_trace(go.Scatter(x=df_temp.time, 
	        						 y=y_truth,
	                                 mode='lines',
                                     name='True',
                                     line=dict(
                                     color=colors[idx])))
	        idx+=1
	        
	    y_pred = df_temp.pred
	    mae_error = masked_mae_loss(torch.tensor(y_pred.values), torch.tensor(y_truth.values))
	    fig.add_trace(go.Scatter(x=df_temp.time, 
	    	                     y=y_pred,
                                 mode='lines',
                                 name='Pred ({} pts, MAE:{:.2f}mph)'.format(h, mae_error),
                                 line=dict(
                                 color=colors[idx])))
	    idx+=1
	    
	 
	fig.update_layout(
	    title='<b>DCRNN Traffic Predictions for Multiple Horizons, PEMS Sensor {}</b>'.format(station),
	#     xaxis_title='<b>Time Stamp</b>',
	    yaxis_title='<b>Average Speed (MPH)</b>',
	#     legend_title="Legend Title",
	    font=dict(size=14, color='white'),
        height=600,
        margin={"r":50,"t":50,"l":50,"b":20},
        template='plotly_dark',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)'
	)

	fig.update_xaxes(rangeslider_visible=True, showgrid=False)
	fig.update_yaxes(showgrid=False)
    
	return fig


@app.callback(
    Output("traffic-animated-plot", "figure"),
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ]
)
def plot_traffic_animated(start_date, end_date):
    
    fig = px.line(traffic_data,  
		         x ='ttime',  
		         y =['aspeed'], 
		         color ='tmonth',
		         line_group = 'tmonth',
		         animation_group ='tmonth', 
		         animation_frame = 'tmonth',
		         labels = {'ttime':'<b>Time of the Day</b>',
		                   'value':'<b>Average Speed</b>',
		                   'tmonth':'<b>Month</b>'},
		         #hover_name ='ttime',  
		         log_y = True,
		         # width = 900,
		         height = 600,
		         range_y = [50,70],
		         #y_title = "Avg Speed by month in 5 min interval',
		         title = "<b>Average Speed by Time of Day in 2020 for San Diego County</b>") 
	
    fig.update_layout(template="plotly_dark", 
    				  plot_bgcolor= 'rgba(0, 0, 0, 0)', 
    				  paper_bgcolor= 'rgba(0, 0, 0, 0)',
    				  font=dict(size=11, color='white'),
				      # height=600,
				      margin={"r":50,"t":30,"l":50,"b":0})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


@app.callback(
    Output("traffic-covid-plot", "figure"),
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ]
)
def plot_traffic_covid(start_date, end_date):

	fig = px.line(traffic_covid_data,  
		             x ='record_date',  
		             y =['norm_min_avg_speed','norm_new_cases'], 
		             #color = 'rmonth',
		             #animation_group ='rmonth', 
		             #animation_frame = 'rmonth',
		             #hover_name ='rmonth', 
		             #log_y = True,
		            labels = {'record_date':'<b>Record Date</b>',
		                       'value':'<b>Normalized Value</b>',
		                       'variable':'<b>Measure</b>'},
		             # width = 1100,
		             height = 600,
		             range_y = [0.01, 1],
		             title = "<b>Normalized Average Speed and Covid Cases by Day for San Diego County</b>") 

	fig.update_layout(
	    margin=dict(t=100, b=70, l=250, r=50),
	    updatemenus=[
	        dict(
	            type="buttons",
	            x=-0.07,
	            y=0.7,
	            showactive=False,
	            buttons=list(
	                [
	                    dict(
	                        label="Both",
	                        method="update",
	                        args=[{"y": [traffic_covid_data["norm_min_avg_speed"], traffic_covid_data["norm_new_cases"]]}],
	                    ),
	                    dict(
	                        label="Normalized Minimum Average Spped",
	                        method="update",
	                        args=[{"y": [traffic_covid_data["norm_min_avg_speed"]]}],
	                    ),
	                    dict(
	                        label="Normalized New Cases",
	                        method="update",
	                        args=[{"y": [traffic_covid_data["norm_new_cases"]]}],
	                    ),
	                    
	                ]),
	        )])
	
	fig.update_layout(template="plotly_dark", plot_bgcolor= 'rgba(0, 0, 0, 0)', paper_bgcolor= 'rgba(0, 0, 0, 0)',
						transition_duration=2000, font=dict(size=14, color='white'), margin={"r":50,"t":50,"l":50,"b":20})
	fig.update_xaxes(showgrid=False)
	fig.update_yaxes(showgrid=False)

	return fig


@app.callback(
    Output("correlations-plot", "figure"),
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ]
)
def plot_correlations(start_date, end_date):
    
    correlations = bay_data.corr(method='pearson')
    
    correlations.new_cases = correlations.new_cases.apply(lambda x: round(x, 6))
    correlations.new_deaths = correlations.new_deaths.apply(lambda x: round(x, 6))
    correlations.tot_total_flow = correlations.tot_total_flow.apply(lambda x: round(x, 6))
    correlations.avg_avg_speed = correlations.avg_avg_speed.apply(lambda x: round(x, 6))
    
    z = correlations.values.tolist()
    x = correlations.columns.tolist()
    y = correlations.index.tolist()
    
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z, colorscale='viridis')

	# for add annotation into Heatmap
    for i in range(len(fig.layout.annotations)):
	    fig.layout.annotations[i].font.size = 12

    fig.update_layout(title_text=f'<b>San Diego County Traffic and Covid Pearson Correlation</b>', 
    				  height=600, template="plotly_dark", 
    				  plot_bgcolor= 'rgba(0, 0, 0, 0)', 
    				  paper_bgcolor= 'rgba(0, 0, 0, 0)',
    				  font=dict(size=14, color='white'),
        			  margin={"r":50,"t":50,"l":50,"b":20})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    return fig



if __name__ == "__main__":
    app.run_server(debug=True)

