"""" this file is created for generating 
    multi chart for plotly"""


def simple_line_chart(df, chart_type_dict, orientation):
    #fetching chart details now
    chart_title = chart_type_dict["title"]
    x_col = chart_type_dict["x_axis"]["column"]
    x_label = chart_type_dict["x_axis"]["label"]
    y_col = chart_type_dict["y_axis"][0]["column"]
    y_label = chart_type_dict["y_axis"][0]["label"]

    chart_data = {
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'chart_title': chart_title,
            'xaxis_title': x_label,
            'yaxis_title': y_label,
            'height': 500,
            'width': 800
        }
    # Converting the extracted data into a list with a structure similar to Plotly Figure
    chart_details = {
                'data': [
                    {
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['yaxis_title'],
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y"], 
                        'yaxis': 'y' 
                    }
                        ],
                'layout': {
                    'height': chart_data['height'],  # Chart height
                    'width': chart_data['width'],  # Chart width
                    'title': {'text': chart_data['chart_title']},  # Chart title
                    'xaxis': {'title': {'text': chart_data['xaxis_title']}},  # X-axis label
                    'yaxis': {'title': {'text': chart_data['yaxis_title']}}  # Y-axis label

                }
            }
    return chart_details


def simple_bar_chart(df, chart_type_dict, orientation):

    #fetching chart details now
    chart_title = chart_type_dict["title"]
    x_col = chart_type_dict["x_axis"]["column"]
    x_label = chart_type_dict["x_axis"]["label"]
    y_col = chart_type_dict["y_axis"][0]["column"]
    y_label = chart_type_dict["y_axis"][0]["label"]

    chart_data = {
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'chart_title': chart_title,
            'xaxis_title': x_label,
            'yaxis_title': y_label,
            'height': 500,
            'width': 800
        }
    # Converting the extracted data into a list with a structure similar to Plotly Figure
    chart_details = {
                'data': [
                    {
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['yaxis_title'],
                        'type': 'bar',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y"], 
                        'yaxis': 'y' 
                    }
                ],
                'layout': {
                    'height': chart_data['height'],  # Chart height
                    'width': chart_data['width'],  # Chart width
                    'title': {'text': chart_data['chart_title']},  # Chart title
                    'xaxis': {'title': {'text': chart_data['xaxis_title']}},  # X-axis label
                    'yaxis': {'title': {'text': chart_data['yaxis_title']}}  # Y-axis label

                }
            }
    return chart_details


def simple_scatter_chart(df, chart_type_dict, orientation):

    #fetching chart details now
    chart_title = chart_type_dict["title"]
    x_col = chart_type_dict["x_axis"]["column"]
    x_label = chart_type_dict["x_axis"]["label"]
    y_col = chart_type_dict["y_axis"][0]["column"]
    y_label = chart_type_dict["y_axis"][0]["label"]

    chart_data = {
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'chart_title': chart_title,
            'xaxis_title': x_label,
            'yaxis_title': y_label,
            'height': 500,
            'width': 800
        }
    # Converting the extracted data into a list with a structure similar to Plotly Figure
    chart_details = {
                'data': [
                    {
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['yaxis_title'],
                        'type': 'scatter',
                        'mode': 'markers',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y"], 
                        'yaxis': 'y' 
                    }
                ],
                'layout': {
                    'height': chart_data['height'],  # Chart height
                    'width': chart_data['width'],  # Chart width
                    'title': {'text': chart_data['chart_title']},  # Chart title
                    'xaxis': {'title': {'text': chart_data['xaxis_title']}},  # X-axis label
                    'yaxis': {'title': {'text': chart_data['yaxis_title']}}  # Y-axis label

                }
            }
    return chart_details


def group_bar_chart(df, chart_type_dict, orientation):

    #fetching chart details now
    chart_title = chart_type_dict["title"]
    x_col = chart_type_dict["x_axis"]["column"]
    x_label = chart_type_dict["x_axis"]["label"]
    y_col = chart_type_dict["y_axis"][0]["column"]
    y_label = chart_type_dict["y_axis"][0]["label"]
    y1_col = chart_type_dict["y_axis"][1]["column"]
    y1_label = chart_type_dict["y_axis"][1]["label"]

    chart_data = {
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'y1': df[y1_col].tolist(),
            'chart_title': chart_title,
            'xaxis_title': x_label,
            'yaxis_title': y_label,
            'y1axis_title': y1_label,
            'height': 500,
            'width': 800
        }
    # Converting the extracted data into a list with a structure similar to Plotly Figure
    chart_details = {
                'data': [
                    {
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['yaxis_title'],
                        'type': 'bar',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y"], 
                        'yaxis': 'y' 
                    },
                    {
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['y1axis_title'],
                        'type': 'bar',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y1"],
                        'yaxis': 'y2' 
                    }
                ],
                'layout': {
                    'barmode': 'group',
                    'height': chart_data['height'],  # Chart height
                    'width': chart_data['width'],  # Chart width
                    'title': {'text': chart_data['chart_title']},  # Chart title
                    'xaxis': {'anchor': 'y', 'domain': [0.0, 0.94], 'title': {'text': chart_data['xaxis_title']}},
                    'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': chart_data['yaxis_title']}},
                    'yaxis2': {'anchor': 'x', 'overlaying': 'y', 'side': 'right', 'title': {'text': chart_data['y1axis_title']}}

                }
            }
    return chart_details


def group_scatter_bar_chart(df, chart_type_dict, orientation):
#fetching chart details now
    chart_title = chart_type_dict["title"]
    x_col = chart_type_dict["x_axis"]["column"]
    x_label = chart_type_dict["x_axis"]["label"]
    y_col = chart_type_dict["y_axis"][0]["column"]
    y_label = chart_type_dict["y_axis"][0]["label"]
    y1_col = chart_type_dict["y_axis"][1]["column"]
    y1_label = chart_type_dict["y_axis"][1]["label"]

    chart_data = {
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'y1': df[y1_col].tolist(),
            'chart_title': chart_title,
            'xaxis_title': x_label,
            'yaxis_title': y_label,
            'y1axis_title': y1_label,
            'height': 500,
            'width': 800
        }
    # Converting the extracted data into a list with a structure similar to Plotly Figure
    chart_details = {
                'data': [
                    {
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['yaxis_title'],
                        'type': 'bar',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y"], 
                        'yaxis': 'y' 
                    },
                    {   
                        'mode': 'lines+markers',
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['y1axis_title'],
                        'type': 'scatter',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y1"],
                        'yaxis': 'y2' 
                    }
                ],
                'layout': {
                    'barmode': 'group',
                    'legend': {'orientation': orientation, 'x': 1, 'xanchor': 'right', 'y': 1.02, 'yanchor': 'bottom'},
                    'height': chart_data['height'],  # Chart height
                    'width': chart_data['width'],  # Chart width
                    'title': {'text': chart_data['chart_title']},  # Chart title
                    'xaxis': {'anchor': 'y', 'domain': [0.0, 0.94], 'title': {'text': chart_data['xaxis_title']}},
                    'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': chart_data['yaxis_title']}},
                    'yaxis2': {'anchor': 'x', 'overlaying': 'y', 'side': 'right', 'title': {'text': chart_data['y1axis_title']}}

                }
            }
    return chart_details


def group_scatter_chart(df, chart_type_dict, orientation):
#fetching chart details now
    chart_title = chart_type_dict["title"]
    x_col = chart_type_dict["x_axis"]["column"]
    x_label = chart_type_dict["x_axis"]["label"]
    y_col = chart_type_dict["y_axis"][0]["column"]
    y_label = chart_type_dict["y_axis"][0]["label"]
    y1_col = chart_type_dict["y_axis"][1]["column"]
    y1_label = chart_type_dict["y_axis"][1]["label"]

    chart_data = {
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'y1': df[y1_col].tolist(),
            'chart_title': chart_title,
            'xaxis_title': x_label,
            'yaxis_title': y_label,
            'y1axis_title': y1_label,
            'height': 500,
            'width': 800
        }
    # Converting the extracted data into a list with a structure similar to Plotly Figure
    chart_details = {
                'data': [
                    {
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['yaxis_title'],
                        'mode': 'lines+markers',
                        'type': 'scatter',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y"], 
                        'yaxis': 'y' 
                    },
                    {   
                        'mode': 'lines+markers',
                        'orientation': orientation,  # Vertical/horizontal  bar chart as per the request
                        'name': chart_data['y1axis_title'],
                        'type': 'scatter',
                        'x': chart_data["x"],  
                        'xaxis': 'x',
                        'y': chart_data["y1"],
                        'yaxis': 'y2' 
                    }
                ],
                'layout': {
                    'barmode': 'group',
                    'legend': {'orientation': orientation, 'x': 1, 'xanchor': 'right', 'y': 1.02, 'yanchor': 'bottom'},
                    'height': chart_data['height'],  # Chart height
                    'width': chart_data['width'],  # Chart width
                    'title': {'text': chart_data['chart_title']},  # Chart title
                    'xaxis': {'anchor': 'y', 'domain': [0.0, 0.94], 'title': {'text': chart_data['xaxis_title']}},
                    'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': chart_data['yaxis_title']}},
                    'yaxis2': {'anchor': 'x', 'overlaying': 'y', 'side': 'right', 'title': {'text': chart_data['y1axis_title']}}

                }
            }
    return chart_details