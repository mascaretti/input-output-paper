import plotly
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def _get_sectors():
    sectors = ['Crop',
           'Forestry',
           'Fishing',
           'Mining',
           'M. food',
           'M. textile',
           'M. wood',
           'M. paper',
           'Print. rec. media',
           'M. coke, petrol.',
           'M. chemicals',
           'M. basic pharma.',
           'M. rubber, plastic prod.',
           'M. mineral prod.',
           'M. basic metals',
           'M. fabr. metal prod.',
           'M. computer prod.',
           'M. electrical equip.',
           'M. machinery',
           'M. MV',
           'M. ot. trans. equip.',
           'M. furniture',
           'Repair equip.',
           'Electr., gas',
           'Water supply',
           'Sewerage',
           'Construction',
           'Wholesale MVs',
           'Wholesale trade',
           'Retail trade',
           'Land trans.',
           'Water trans.',
           'Air trans.',
           'Warehousing',
           'Postal act.',
           'Accommodations',
           'Publish. act.',
           'Motion picture',
           'Telcom',
           'Computer progr.',
           'Financial service',
           'Insurance, pension',
           'Aux. to finan. services',
           'Real estate',
           'Legal',
           'Arch., engin.',
           'Scientific R&D',
           'Ads and market',
           'Oth. prof. act.',
           'Admin. act.',
           'PA and defence',
           'Education',
           'Human health',
           'Other service']
    return sectors

def barplot(centrality_values, title, colour, file_dir, sectors=None):
    '''This function plots a barplot containing a centrality value
    for each sector. The colour is specified by the user.
    Plot is saved in HTML format.
    '''
    sectors = _get_sectors() if sectors is None else sectors
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sectors,
        y=list(centrality_values),
        marker=dict(
            color=colour,
        )
    ))

    fig.update_layout(
        title=title)

    plotly.io.write_html(fig, file=file_dir + "html")
    fig.write_image(file_dir + "pdf", width = 1800, height = 950)

def lineplot(input_values, name_list, title, xaxis_title, yaxis_title, file_dir):

    try:
        assert type(input_values) is list
    except AssertionError:
        input_values = [input_values]

    try:
        assert type(name_list) is list
    except AssertionError:
        name_list = [name_list]

    try:
        assert len(name_list) == len(input_values)
    except AssertionError:
        print("\nLength of input values must match length of input names")
        raise

    fig = go.Figure()

    for i, value in enumerate(input_values):
        fig.add_trace(go.Scatter(x=value[0],y=value[1],
                    mode='lines+markers',
                    name=name_list[i]))

        fig.update_layout(title=title,
                   xaxis_title=xaxis_title,
                   yaxis_title=yaxis_title
                 )
    
    plotly.io.write_html(fig, file=file_dir + "html")
    fig.write_image(file_dir + "pdf")


def heatmap(A, title, file_dir):
    n = A.shape[0]
    labels = [j for j in range(1, n + 1)]
    fig = go.Figure(data=go.Heatmap(
                   z=A,
                   x=labels,
                   y=labels,
                   ))
    fig.update_layout(title=title, xaxis=dict(tickmode = 'linear'), yaxis=dict(tickmode = 'linear'))
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    plotly.io.write_html(fig, file=file_dir + "html")
    fig.write_image(file_dir + "pdf", width=700, height=700)


def radar(centrality_values, colour, title, file_dir, num_sectors=5):
    df = pd.DataFrame(dict(
    r=centrality_values,
    theta=_get_sectors()))

    df = df.sort_values(by=['r'])[:num_sectors]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=df['r'],
        theta=df['theta'],
        fill='toself',
        fillcolor=colour,
        line=
        dict(color=colour),
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True
        ),
    ),
    title=title
    )
    plotly.io.write_html(fig, file=file_dir + "html")
    fig.write_image(file_dir + "pdf", width=1200, height=600)

