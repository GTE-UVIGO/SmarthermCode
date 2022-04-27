import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import plotly.subplots as subplots
import os


def minmax(pathname):
    filenames = []
    # names = []

    for i in os.listdir(pathname):
        # names.append(i[7:12])
        filenames.append(i)

    T = []
    for n, i in enumerate(filenames):
        temp = pd.read_csv(pathname + "/" + i, sep=',', index_col=0)
        temp = temp.to_numpy()
        T.append(temp)

    zmin = np.min(T)
    zmax = np.max(T)

    return zmin, zmax



def plot(fig, filePath, gridsize, title, legend, row, col, num, showscale, zmin, zmax, ypos, name, colorscale, reversescale=False):
    temp = pd.read_csv(filePath, sep=',', index_col=0)
    # temp = temp[::-1]
    # temp = np.fliplr(temp)
    #name = filePath[-11:-6]
    # print(name)

    x = np.array(range(temp.shape[1])) * gridsize
    y = np.array(range(temp.shape[0])) * gridsize

    # zmin = np.min(temp)
    # zmax = np.max(temp)
    y[-1] = 16.3
    x[-1] = 50.6

    trace1 = go.Contour(z=temp,
                        x=x,
                        y=y,
                        zmin=zmin,
                        zmax=zmax,
                        #line=dict(width=1.15),
                        # x0=0,
                        # y0=0,
                        # dx=10,
                        # dy=10,
                        # colorbar=dict(scaleanchor='x'),
                        contours_coloring='heatmap',
                        colorscale=colorscale,
                        # colorbar=dict(len=0.90, x=ypos, thickness=8, title=dict(text=legend, side='right', font=dict(size=15)),
                        colorbar=dict(len=0.81, x=ypos, thickness=16, title=dict(text=legend, side='right', font=dict(size=25)),
                                      tickfont=dict(size=20), tickangle=-90),
                        reversescale=reversescale,
                        showscale=showscale,
                        )


    trace2 = go.Contour(z=[0, 0, 0, 0],
                        x=[4.64, 9.04, 4.64, 9.04],
                        y=[5.92, 5.92, 10.95, 10.95],
                        colorscale=[[0, 'white'], [1, 'white']],
                        showlegend=False,
                        showscale=False)

    trace3 = go.Contour(z=[0, 0, 0, 0],
                        x=[11.45, 16.05, 16.05, 11.45],
                        y=[3.1, 3.1, 13.1, 13.1],
                        colorscale=[[0, 'white'], [1, 'white']],
                        showlegend=False,
                        showscale=False)

    trace4 = go.Contour(z=[0, 0, 0, 0],
                        x=[13.55, 16.05, 16.05, 13.55],
                        y=[12.9, 12.9, 16.35, 16.35],
                        colorscale=[[0, 'white'], [1, 'white']],
                        showlegend=False,
                        showscale=False)

    trace5 = go.Contour(z=[0, 0, 0, 0],
                        x=[30.7, 39.6, 39.6, 30.7],
                        y=[6.85, 6.85, 16.35, 16.35],
                        colorscale=[[0, 'white'], [1, 'white']],
                        showlegend=False,
                        showscale=False)

    trace6 = go.Contour(z=[0, 0, 0, 0],
                        x=[36.45, 43.95, 43.95, 36.45],
                        y=[3.5, 3.5, 10.8, 10.8],
                        colorscale=[[0, 'white'], [1, 'white']],
                        showlegend=False,
                        showscale=False)

    trace7 = go.Contour(z=[0, 0, 0, 0],
                        x=[39.5, 41.7, 41.7, 39.5],
                        y=[14, 14, 16.35, 16.35],
                        colorscale=[[0, 'white'], [1, 'white']],
                        showlegend=False,
                        showscale=False)

    trace8 = go.Contour(z=[0, 0, 0, 0],
                        x=[41.65, 43.85, 43.85, 41.65],
                        y=[13.4, 13.4, 16.35, 16.35],
                        colorscale=[[0, 'white'], [1, 'white']],
                        showlegend=False,
                        showscale=False)

    trace9 = go.Contour(z=[0, 0, 0, 0],
                        x=[47.25, 50.6, 50.6, 47.25],
                        y=[3.1, 3.1, 16.35, 16.35],
                        colorscale=[[0, 'white'], [1, 'white']],
                        showlegend=False,
                        showscale=False)

    # layout = go.Layout(autosize=True,
    #                    # width=1044,
    #                    # height=300,
    #                    margin=dict(r=5, l=5, b=60, t=10),
    #                    plot_bgcolor='rgba(0,0,0,0)',
                       # title="Temperatura - 25/09/2020 18:10",
                       # title="Concentración CO<sub>2</sub> - 25/09/2020 18:10",
                       # title="Humedad Relativa - 25/09/2020 18:10",
                       # yaxis=dict(showline=False, showgrid=False, visible=False, scaleanchor="x", scaleratio=1, constrain="domain", zeroline=False, range=[-1, 16.5]),
                       # xaxis=dict(showline=False, showgrid=False, visible=False, linecolor='black', zeroline=False, range=[-1, 51]),
                       # images=[dict(source='https://raw.githubusercontent.com/aogando/Farola/master/PlanoMTI.png',
                       #                      xref='x domain', yref='y domain', y=0.533, x=0.503, sizex=1.01, sizey=1.01, xanchor='center', yanchor='middle')]
                       # )

    # fig = go.Figure()
    fig.add_traces([trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9], cols=col, rows=row)
    #fig.add_traces([trace1], cols=col, rows=row)

    if num == 3 or num== 5:
        fig.add_annotation(dict(font=dict(size=25), text=str(name[0:2]) + ':' + str(name[3:]), xref='x' + str(num) + ' domain',
                                yref='y' + str(num) + ' domain', x=-0.015, y=0.5, showarrow=False, textangle=-90))

    if num == 2:
        fig.add_annotation(dict(font=dict(size=30), text=title, xref='x' + str(num) + ' domain', yref='y' + str(num) + ' domain', x=0.5, y=1.35, showarrow=False))


    if num == 1:
        fig.update_layout({'xaxis': {'showline': False, 'showgrid': False, 'visible': False, 'scaleratio': 1,
                                                'scaleanchor': 'x'}})
        fig.update_layout({'yaxis': {'showline': False, 'showgrid': False, 'visible': False, 'scaleratio': 1,
                                                'scaleanchor': 'x'}})

        fig.add_layout_image(dict(source='https://raw.githubusercontent.com/aogando/Farola/master/PlanoMTI_3.png',
                                  xref='x domain', yref='y domain', y=0.512, x=0.496,
                                  sizex=1.155, sizey=1.155, xanchor='center', yanchor='middle'))

        fig.add_annotation(dict(font=dict(size=25), text=str(name[0:2]) + ':' + str(name[3:]), xref='x domain',
                                yref='y domain', x=-0.015, y=0.5, showarrow=False, textangle=-90))

        fig.add_annotation(dict(font=dict(size=30), text=title, xref='x domain', yref='y domain', x=0.5, y=1.35, showarrow=False))

    else:
        fig.update_layout({'xaxis' + str(num): {'showline': False, 'showgrid': False, 'visible': False, 'scaleratio': 1,
                                                'scaleanchor': 'x'}})
        fig.update_layout({'yaxis' + str(num): {'showline': False, 'showgrid': False, 'visible': False, 'scaleratio': 1,
                                                'scaleanchor': 'x'}})

        fig.add_layout_image(dict(source='https://raw.githubusercontent.com/aogando/Farola/master/PlanoMTI_3.png',
                                  xref='x' + str(num) + ' domain', yref='y' + str(num) + ' domain', y=0.512, x=0.496,
                                  sizex=1.155, sizey=1.155, xanchor='center', yanchor='middle'))


zminT, zmaxT = minmax("E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/temp")
zminHR, zmaxHR = minmax("E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/hr")

fig = subplots.make_subplots(rows=2, cols=2, vertical_spacing=0.050, horizontal_spacing=0.015)
layout = go.Layout(autosize=False,
                   width=1525,
                   height=936,
                   margin=dict(r=4, l=50, b=150, t=80),
                   plot_bgcolor='rgba(0,0,0,0)',
                   )
fig.update_layout(layout)


blues__ = [[0, 'rgb(247,247,247)'],
           [0.2, 'rgb(209,229,240)'],
           [0.4, 'rgb(146,197,222)'],
           [0.6, 'rgb(67,147,195)'],
           [0.8, 'rgb(33,102,172)'],
           [1, 'rgb(5,48,97)']]

greens__ = [[0, 'rgb(247,247,247)'],
            [0.2, 'rgb(217,240,211)'],
            [0.4, 'rgb(166,219,160)'],
            [0.6, 'rgb(90,174,97)'],
            [0.8, 'rgb(27,120,55)'],
            [1, 'rgb(0,68,27)']]

reds__ = [[0, 'rgb(103,0,31)'],
          [0.2, 'rgb(178,24,43)'],
          [0.4, 'rgb(214,96,77)'],
          [0.6, 'rgb(244,165,130)'],
          [0.8, 'rgb(253,219,199)'],
          [1, 'rgb(247,247,247)']]

aaa = [[0, 'rgb(103,0,31)'],
       [0.1, 'rgb(178,24,43)'],
       [0.2, 'rgb(214,96,77)'],
       [0.3, 'rgb(244,165,130)'],
       [0.4, 'rgb(253,219,199)'],
       [0.5, 'rgb(247,247,247)'],
       [0.6, 'rgb(209,229,240)'],
       [0.7, 'rgb(146,197,222)'],
       [0.8, 'rgb(67,147,195)'],
       [0.9, 'rgb(33,102,172)'],
       [1, 'rgb(5,48,97)']]

bbb = [[0, 'rgb(103,0,31)'],
       [0.1, 'rgb(178,24,43)'],
       [0.2, 'rgb(214,96,77)'],
       [0.3, 'rgb(244,165,130)'],
       [0.4, 'rgb(253,219,199)'],
       [0.5, 'rgb(247,247,247)'],
       [0.6, 'rgb(180, 217, 204)'],
       [0.7, 'rgb(137, 192, 182)'],
       [0.8, 'rgb(68, 140, 138)'],
       [0.9, 'rgb(40, 114, 116)'],
       [1, 'rgb(13, 88, 95)']]

ccc = [[0, 'rgb(103,0,31)'],
       [0.1, 'rgb(178,24,43)'],
       [0.2, 'rgb(214,96,77)'],
       [0.3, 'rgb(244,165,130)'],
       [0.4, 'rgb(253,219,199)'],
       [0.5, 'rgb(247,247,247)'],
       [0.6, 'rgb(255,255,204)'],
       [0.7, 'rgb(255,237,160)'],
       [0.8, 'rgb(254,217,118)'],
       [0.9, 'rgb(254,178,76)'],
       [1, 'rgb(253,141,60)']]
ccc2 = [[0, 'rgb(103,0,31)'],
       [0.1, 'rgb(178,24,43)'],
       [0.2, 'rgb(214,96,77)'],
       [0.3, 'rgb(244,165,130)'],
       [0.4, 'rgb(253,219,199)'],
       [0.5, 'rgb(247,247,247)'],
       [0.6, 'rgb(246,232,195)'],
       [0.7, 'rgb(223,194,125)'],
       [0.8, 'rgb(191,129,45)'],
       [0.9, 'rgb(140,81,10)'],
       [1, 'rgb(84,48,5)']]

ccc3 = [[0, 'rgb(84,48,5)'],
       [0.1, 'rgb(140,81,10)'],
       [0.2, 'rgb(191,129,45)'],
       [0.3, 'rgb(223,194,125)'],
       [0.4, 'rgb(246,232,195)'],
       [0.5, 'rgb(247,247,247)'],
       [0.6, 'rgb(216,218,235)'],
       [0.7, 'rgb(178,171,210)'],
       [0.8, 'rgb(128,115,172)'],
       [0.9, 'rgb(84,39,136)'],
       [1, 'rgb(45,0,75)']]



#plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/temp/['temp12.3'].csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=1, col=1, num=1, showscale=True, zmin=zminT, zmax=zmaxT, ypos=0.84,name='12.30', colorscale=aaa, reversescale=True)
#plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/temp/['temp13.0'].csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=2, col=1, num=3, showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.84, name='13.00',colorscale=aaa, reversescale=True)
#plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/temp/['temp13.3'].csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=3, col=1, num=5, showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.84, name='13.30',colorscale=aaa, reversescale=True)
#plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/hr/['hum12.3'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=1, col=2, num=2, showscale=True, zmin=zminHR, zmax=zmaxHR, ypos=0.84, name='12.30',colorscale='PuOr', reversescale=False)
#plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/hr/['hum13.0'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=2, col=2, num=4, showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.84, name='13.00',colorscale='PuOr', reversescale=False)
#plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/hr/['hum13.3'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=3, col=2, num=6, showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.84, name='13.30',colorscale='PuOr', reversescale=False)

plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/temp/['temp12.3'].csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=1, col=1, num=1, showscale=True, zmin=zminT, zmax=zmaxT, ypos=0.84,name='12.30', colorscale=aaa, reversescale=True)
#plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/temp/['temp13.0'].csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=2, col=1, num=3, showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.84, name='13.00',colorscale=aaa, reversescale=True)
plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/temp/['temp13.3'].csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=2, col=1, num=5, showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.84, name='13.30',colorscale=aaa, reversescale=True)
plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/hr/['hum12.3'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=1, col=2, num=2, showscale=True, zmin=zminHR, zmax=zmaxHR, ypos=0.84, name='12.30',colorscale=ccc3, reversescale=False)
#plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/hr/['hum13.0'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=2, col=2, num=4, showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.84, name='13.00',colorscale=ccc3, reversescale=False)
plot(fig, "E:\Documents\Doctorado\CNIT_2021\Ponencia_interpolation/hr/['hum13.3'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=2, col=2, num=6, showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.84, name='13.30',colorscale=ccc3, reversescale=False)


plotly.offline.plot(fig,
                    auto_open=True,
image='svg',
                    image_filename='Plot',
                    image_width=1525,
                    image_height=936,)

