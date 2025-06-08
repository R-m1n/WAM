import io
import base64
from pathlib import Path

import numpy as np
import pandas as pd

import dash
from dash import (
    Dash, 
    dcc,
    html, 
    dash_table, 
    Input, 
    Output, 
    State,
)

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

from dea_model import dea_one_model

from reportlab.platypus import (
    SimpleDocTemplate, 
    Paragraph, 
    Spacer, 
    Table, 
    TableStyle, 
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A3
from reportlab.lib import colors


script_dir = Path(__file__).resolve().parent
input_path = script_dir / "input_data.xlsx"

initial_df = pd.read_excel(input_path)


DARKBLUE = "#1c2b4d"
WHITE = "#ffffff"
GREY = "#dbdada"
GREEN_HIGHLIGHT = "#e0ffe0"
YELLOW_HIGHLIGHT = "#fdffe0"
RED_HIGHLIGHT = "#ffe0e0"

SUCCESS = "✅"
FAIL = "❌"

SEQ_COLORS = px.colors.sequential.Turbo

FONT = "Georgia"
ANOTHER_FONT = "Courier New"


table_style = {
    "overflowX": "auto",
    "borderRadius": "30px",
    "border": "4px solid #ccc",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.4)",
}

green_query, yellow_query, red_query = (
    f"{{Efficiency Score}} >= {0.9}", 
    f"{{Efficiency Score}} >= {0.8} && {{Efficiency Score}} < {0.9}", 
    f"{{Efficiency Score}} < {0.8}",
)

row_highlight_conditionals = [
    {
        "if": {"filter_query": green_query}, 
        "backgroundColor": GREEN_HIGHLIGHT,
    },
    {
        "if": {"filter_query": yellow_query},
        "backgroundColor": YELLOW_HIGHLIGHT,
    },
    {
        "if": {"filter_query": red_query}, 
        "backgroundColor": RED_HIGHLIGHT,
    },
]

layout = html.Div([
    html.H1("One-Model DEA Input Congestion Dashboard", 
            style={
                "textAlign": "center",
                "fontFamily": FONT,
            },
            className="fancy-header",
        ),

    html.Div([
        html.H3(
            "Input Data Table",
            style={
                "fontWeight": "bold",
                "fontSize": "24px",
                "fontFamily": FONT,
            },
            className="fancy-subheader",
        ),
        dash_table.DataTable(
            id="input-table",
            data=initial_df.to_dict("records"),
            columns=[
                {
                    "name": col, 
                    "id": col,
                    "editable": (col != "DMU"),
                } 
                for col in initial_df.columns
            ],
            style_header={
                "backgroundColor": DARKBLUE,
                "fontColor": WHITE,
                "color": WHITE,
                "fontWeight": "bold",
            },
            style_cell={
                "textAlign": "center",
                "fontWeight": "bold",
                "fontFamily": ANOTHER_FONT,
                "border": "2px solid #ccc",
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'University'},
                    'width': '250px'
                },
            ],
            editable=True,
            row_deletable=False,
            style_table=table_style,
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": GREY,
                },
                {
                    "if": {"column_editable": False},
                    "backgroundColor": DARKBLUE,
                    "color": WHITE,
                    "fontWeight": "bold",
                },
            ],
            page_size=12,
        ),
        html.Div([
            html.Div([
                html.Button(
                    "🔁 Update DEA", 
                    id="run-dea", 
                    n_clicks=0, 
                    className="custom-button",
                )
            ],
            ),
            html.Div([
                dcc.Upload(
                    id="upload-excel",
                    children=html.Button(
                        "📤 Import Excel", 
                        className="custom-button",
                    ),
                    accept=".xlsx",
                    multiple=False,
                )
            ],
            ),
            html.Div([
                html.Button(
                    "📥 Export Excel", 
                    id="download-input-btn", 
                    className="custom-button",
                ),
                dcc.Download(id="excel-input-download")
            ],
            ),
            html.Div([
                dcc.Loading(
                    type="default",
                    children=html.Div(id="dea-status"),
                    className="custom-spinner",
                ),
            ],
            style={
                "minWidth": "80px",
            },
            ),
        ], 
        style={
            "display": "flex",
            "margin": "20px",
            "gap": "20px",
            "alignItems": "center",
        },
    ),
    ],
    style={
        "marginBottom": "30px",
        }
    ),
    html.Hr(),
    html.Div([
        dcc.Dropdown(
            options=[
                {"label": u, "value": u}
                for u in initial_df["University"]
            ],
            id="university-dropdown",
            multi=True,
            value=initial_df["University"].tolist(),
            placeholder="Select universities...",
            style={
                "width": "100%",
                "fontFamily": ANOTHER_FONT,
                "fontWeight": "bold",
                "borderRadius": "5px",
                "border": "1px solid #ccc",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
            },
        ),
        ], 
        style={
            "display": "flex",
            "justifyContent": "center",
            "maxWidth": "1200px",
            "width": "100%",
            "margin": "0 auto",
            "marginTop": "20px",
            "marginBottom": "30px",
        },
    ),
    dcc.Checklist(
        options=[
            {
                "label": "ٍEfficiency Score", 
                "value": "Efficiency Score",
            },
            {
                "label": "Congestion", 
                "value": "Congestion",
            },
            {
                "label": "Extra Faculty Needed", 
                "value": "Extra Faculty Needed",
            },
            {
                "label": "Citation Slack", 
                "value": "Citation Slack",
            },
            {
                "label": "Paper Slack", 
                "value": "Paper Slack",
            },
        ],
        value=[
            "Efficiency Score", 
            "Congestion", 
            "Extra Faculty Needed", 
            "Citation Slack",
        ],
        id="metric-checklist",
        inline=True,
        style={
            "textAlign": "center",
        },
        labelStyle={
            "display": "inline-block",
            "borderRadius": "40px",
            "border": "1px solid #ccc",
            "padding": "10px",
            "marginRight": "20px",
            "backgroundColor": GREY,
            "fontFamily": FONT,
        },
    ),

    html.Div(id="metric-plots"),

    html.Hr(),
    dcc.Tabs([
        dcc.Tab(
            label="Parallel Coordinates Plot", 
            children=[
                html.Div([
                    dcc.Graph(id="parallel-coord"),
                    ],
                    style={
                        "marginTop": "40px",   
                    }
                ),
            ],
        ),
        dcc.Tab(label="Correlation Heatmap", 
            children=[
                html.Div([
                        dcc.Graph(id="corr-heatmap"),
                    ], 
                    style={
                        "textAlign": "center", 
                        "padding": "20px",
                    },
                ),
            ],
        ),
    ], 
    style={
        "paddingTop": "20px",
        "fontFamily": FONT,
    }
    ),

    html.H3(
        "Results Table",
        style={
            "fontWeight": "bold",
            "fontSize": "24px",
            "fontFamily": FONT,
        },
        className="fancy-subheader",
    ),

    dcc.RadioItems(
        options=[
            {
                "label": "DMU", 
                "value": "DMU",
            },
            {
                "label": "Efficiency Score", 
                "value": "Efficiency Score",
            },
            {
                "label": "Congestion", 
                "value": "Congestion",
            },
            {
                "label": "Extra Faculty Needed", 
                "value": "Extra Faculty Needed",
            },
            {
                "label": "Citation Slack", 
                "value": "Citation Slack",
            },
            {
                "label": "Paper Slack", 
                "value": "Paper Slack",
            },
        ],
        value="Efficiency Score",
        id="order-option",
        inline=True,
        style={
            "paddingBottom": "20px",
            "textAlign": "center",
        },
        labelStyle={
            "display": "inline-block",
            "paddingRight": "20px",
            "fontFamily": FONT,
            "borderRadius": "40px",
            "border": "1px solid #ccc",
            "padding": "10px",
            "marginRight": "20px",
            "backgroundColor": GREY,
        },
    ),

    dash_table.DataTable(
        id="results-table",
        columns=[],
        data=[],
        style_table={
            **table_style,
            "marginBottom": "10px",
        },
        page_size=12,
        style_header={
            "backgroundColor": DARKBLUE,
            "fontColor": WHITE,
            "color": WHITE,
            "fontWeight": "bold",
        },
        style_cell={
            "textAlign": "center",
            "fontWeight": "bold",
            "fontFamily": ANOTHER_FONT,
            "border": "2px solid #ccc",
        },
        style_data_conditional=row_highlight_conditionals,
    ),

    html.Div([
        html.Div([
            html.Button(
                "📥 Export Excel", 
                id="download-excel-btn", 
                className="custom-button",
            ),
            dcc.Download(id="excel-download"),
        ],
        ),

        html.Div([
            html.Button(
                "📋 Download Report", 
                id="download-report-btn", 
                className="custom-button",
            ),
            dcc.Download(id="report-download"),
        ],
        ),

        html.Div([
            dcc.Loading(
                type="default",
                children=html.Div(id="progress-status"),
                className="custom-spinner",
            ),
        ],
        style={
            "minWidth": "80px",
        },
        ),
    ], 
    style={
        "display": "flex",
        "marginTop": "20px",
        "marginBottom": "100px",
        "gap": "20px",
        "alignItems": "center",
    },
    ),

    dcc.Store(id="dea-data"),
    dcc.Store(id="stored-metric-figures"),
],

style={
    "paddingTop": "50px",
    "paddingLeft": "200px",
    "paddingRight": "200px",
},
)


assets_dir = script_dir / "assets"

app = Dash(
    __name__,
    assets_folder=assets_dir,
)

app.title = "One-Model DEA Input Congestion Dashboard"

app.layout = layout


@app.callback(
    Output("results-table", "columns"),
    Output("results-table", "data"),
    Output("dea-data", "data"),
    Output("dea-status", "children"),
    Input("run-dea", "n_clicks"),
    Input("order-option", "value"),
    State("input-table", "data"),
)
def run_dea(n_clicks, order, table_data):
    try:
        df = pd.DataFrame(table_data)
        X = df[["Faculty"]].T.values
        Y = df[["Citation", "Paper"]].T.values

        results = []
        for i in range(X.shape[1]):
            res = dea_one_model(X, Y, i)
            results.append(
                {
                    "University": df.loc[i, "University"],
                    "phi": res["phi"],
                    "Congestion": res["s_c"][0],
                    "Extra Faculty Needed": res["s_plus_i2"][0],
                    "Citation Slack": res["s_plus_r"][0],
                    "Paper Slack": res["s_plus_r"][1],
                }
            )

        result_df = pd.DataFrame(results)

        result_df["phi"] = (
            result_df["phi"]
                .astype(np.float64)
        )

        result_df["Congestion"] = (
            result_df["Congestion"]
                .astype(np.int32)
        )

        result_df["Extra Faculty Needed"] = (
            result_df["Extra Faculty Needed"]
                .astype(np.int32)
        )

        result_df["Citation Slack"] = (
            result_df["Citation Slack"]
                .astype(np.float64)
                .round(2)
        )

        result_df["Paper Slack"] = (
            result_df["Paper Slack"]
                .astype(np.float64)
                .round(2)
        )

        result_df["phi"] = (result_df["phi"].max() - result_df["phi"] + 1)
        
        result_df["phi"] = (
            MinMaxScaler().fit_transform(
                result_df[["phi"]].to_numpy()
            )
        )
        
        result_df["phi"] = result_df["phi"].round(2)

        result_df.rename(
            columns={
                "phi": "Efficiency Score"
            }, 
            inplace=True,
        )

        result_df.index += 1

        result_df.index.name = "DMU"

        result_df.reset_index(inplace=True)

        result_df.sort_values(
            by=order,
            ascending=(order == "DMU"), 
            inplace=True,
        )

        columns = [
            {
                "name": col, 
                "id": col,
            } 
            for col in result_df.columns
        ]

        return (
            columns, 
            result_df.to_dict("records"),
            result_df.to_dict("records"),
            SUCCESS * bool(n_clicks),
        )
    
    except Exception as e:
        return (
            dash.no_update, 
            dash.no_update,
            dash.no_update,
            FAIL + f" {e}",
        )


@app.callback(
    Output("input-table", "data"),
    Output("input-table", "columns"),
    Input("upload-excel", "contents"),
    State("upload-excel", "filename"),
    prevent_initial_call=True
)
def update_table_from_excel(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if filename.endswith(".xlsx"):
        df_uploaded = pd.read_excel(io.BytesIO(decoded))
    else:
        return dash.no_update, dash.no_update

    return (
        df_uploaded.to_dict("records"), 
        [
            {
                "name": col, 
                "id": col, 
                "editable": col != "DMU"
            } 
            for col in df_uploaded.columns
        ]
    )


@app.callback(
    Output("excel-input-download", "data"),
    Input("download-input-btn", "n_clicks"),
    State("input-table", "data"),
    prevent_initial_call=True
)
def download_excel_input(_, data):
    return dcc.send_data_frame(
        pd.DataFrame(data).to_excel, 
        "input_data.xlsx",
        index=False,
    )


@app.callback(
    Output("university-dropdown", "options"),
    Output("university-dropdown", "value"),
    Input("run-dea", "n_clicks"),
    State("input-table", "data"),
)
def update_uni_dropdown(_, table_data):
    df = pd.DataFrame(table_data)

    return (
        [
            {"label": u, "value": u}
            for u in df["University"]
        ], 
        df["University"].to_list()
    )


@app.callback(
    Output("metric-plots", "children"),
    Output("stored-metric-figures", "data"),
    Input("university-dropdown", "value"),
    Input("metric-checklist", "value"),
    Input("dea-data", "data"),
)
def update_metric_charts(selected_uni, selected_metrics, data):
    if not (data and selected_uni and selected_metrics):
        return [], []

    df = pd.DataFrame(data)

    df.sort_values(
        by="Efficiency Score", 
        ascending=False, 
        inplace=True,
    )

    df = df[df["University"].isin(selected_uni)]

    plots = []
    figures = []
    for metric in selected_metrics:
        fig = px.bar(
            df,
            x="University",
            y=metric,
            color="University",
            color_discrete_sequence=SEQ_COLORS,
        )
        figures.append(fig.to_dict())

        plots.append(
            html.Div(
                children=dcc.Graph(figure=fig),
                style={
                    "width": "48%",
                    "display": "inline-block",
                    "verticalAlign": "top",
                    "padding": "1%",
                }
            )
        )

    return plots, figures


@app.callback(
    Output("parallel-coord", "figure"),
    Input("dea-data", "data"),
)
def update_parallel_coord(data):
    if not data:
        return go.Figure()

    df = pd.DataFrame(data)

    df.sort_values(
        by="Efficiency Score", 
        ascending=False, 
        inplace=True,
    )

    metrics = [
        "Efficiency Score", 
        "Congestion", 
        "Extra Faculty Needed", 
        "Citation Slack", 
        "Paper Slack",
    ]

    df[metrics] = MinMaxScaler().fit_transform(df[metrics])

    uni_codes, uni_labels = pd.factorize(df["University"])

    parcoords = go.Parcoords(
        line={
            "color": uni_codes, 
            "colorscale": SEQ_COLORS, 
            "showscale": False
        },

        dimensions=[
            {
                "label": col, 
                "values": df[col],
            } 
            for col in metrics
        ],
    )

    legend_items = [
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker={
                "size": 10, 
                "color": SEQ_COLORS[code % len(SEQ_COLORS)]
            },
            name=label,
        )
        for code, label in zip(uni_codes, uni_labels)
    ]

    fig = go.Figure(
        data=[parcoords] + legend_items
    )

    fig.update_layout(
        legend_title="University",
        plot_bgcolor="white",
        xaxis={
            "showgrid": False, 
            "showticklabels": False, 
            "zeroline": False
        },
        yaxis={
            "showgrid": False, 
            "showticklabels": False, 
            "zeroline": False
        },
    )

    return fig


@app.callback(
    Output("corr-heatmap", "figure"),
    Input("dea-data", "data"),
)
def update_corr_heatmap(data):
    if not data:
        return None
    
    df = pd.DataFrame(data)

    metrics = [
        "Efficiency Score", 
        "Congestion", 
        "Extra Faculty Needed", 
        "Citation Slack", 
        "Paper Slack"
    ]

    corr_matrix = df[metrics].corr()

    heatmap = go.Heatmap(
        x=metrics,
        y=metrics,
        z=corr_matrix,
        text=corr_matrix,
        type = "heatmap",
        hoverongaps = False,
        colorscale="RdBu_r",
        texttemplate="%{text:.2f}"
    )

    fig = go.Figure(data=heatmap)

    return fig


@app.callback(
    Output("excel-download", "data"),
    Output("progress-status", "children", allow_duplicate=True),
    Input("download-excel-btn", "n_clicks"),
    State("dea-data", "data"),
    prevent_initial_call=True,
)
def download_excel(_, data):
    try:
        return (
            dcc.send_data_frame(
                pd.DataFrame(data).to_excel, 
                "dea_results.xlsx",
                index=False,
            ), 
            SUCCESS,
        )
    
    except Exception as e:
        return dash.no_update, FAIL + f" {e}"


@app.callback(
    Output("report-download", "data"),
    Output("progress-status", "children", allow_duplicate=True),
    Input("download-report-btn", "n_clicks"),
    State("input-table", "data"),
    State("stored-metric-figures", "data"),
    State("parallel-coord", "figure"),
    State("corr-heatmap", "figure"),
    State("results-table", "data"),
    prevent_initial_call=True
)
def generate_pdf_report(
    _, 
    input_table, 
    metric_figures, 
    parallel_coord, 
    corr_heatmap, 
    results_table
):
    try:
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(buffer, pagesize=A3)
        story = []
        styles = getSampleStyleSheet()

        story.append(Paragraph("DEA Input Congestion Report", styles['Title']))
        story.append(Spacer(1, 40))

        story.append(Paragraph("Input Table", styles['Heading2']))
        story.append(Spacer(1, 20))
        story.append(table_to_reportlab(input_table))
        story.append(Spacer(1, 100))

        story.append(Paragraph("Results Table", styles['Heading2']))
        story.append(Spacer(1, 20))
        story.append(table_to_reportlab(results_table))
        story.append(Spacer(1, 12))

        for fig_dict in metric_figures:
            fig = go.Figure(fig_dict)
            story.append(plot_to_image(fig))
            story.append(Spacer(1, 6))

        story.append(plot_to_image(parallel_coord))
        story.append(plot_to_image(corr_heatmap))
        story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)

        return dcc.send_bytes(buffer.read(), filename="dea_report.pdf"), SUCCESS
    
    except Exception as e:
        return dash.no_update, FAIL + f" {e}"


def table_to_reportlab(data):
    if not data:
        return Paragraph("No data available", getSampleStyleSheet()['Normal'])

    df = pd.DataFrame(data)
    table_data = [df.columns.tolist()] + df.values.tolist()

    tbl = Table(table_data, repeatRows=1)

    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))

    return tbl


def plot_to_image(plot):
    if not plot:
        return RLImage(None)

    width = 800
    height = 500

    img_bytes = pio.to_image(
        plot, 
        format="png",
        width=width,
        height=height,
        scale=2,
    )

    img_buf = io.BytesIO(img_bytes)
    img_buf.seek(0)

    return RLImage(
        img_buf, 
        width=width, 
        height=height,
    )


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=8080)