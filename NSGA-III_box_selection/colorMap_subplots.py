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


def plot(fig, filePath, gridsize, title, legend, row, col, num, showscale, zmin, zmax, ypos):
    temp = pd.read_csv(filePath, sep=',', index_col=0)
    # temp = temp[::-1]
    # temp = np.fliplr(temp)
    name = filePath[-11:-6]
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
                        # x0=0,
                        # y0=0,
                        # dx=10,
                        # dy=10,
                        # colorbar=dict(scaleanchor='x'),
                        colorscale='rdbu',
                        # colorbar=dict(len=0.90, x=ypos, thickness=8, title=dict(text=legend, side='right', font=dict(size=15)),
                        colorbar=dict(len=0.70, x=ypos, thickness=8,
                                      title=dict(text=legend, side='right', font=dict(size=15)),
                                      tickfont=dict(size=13), tickangle=-90),
                        reversescale=True,
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
    fig.add_traces([trace1], cols=col, rows=row)

    if num == 4 or num == 7:
        fig.add_annotation(
            dict(font=dict(size=25), text=str(name[0:2]) + ':' + str(name[3:]), xref='x' + str(num) + ' domain',
                 yref='y' + str(num) + ' domain', x=-0.09, y=0.5, showarrow=False, textangle=-90))

    if num == 2 or num == 3:
        fig.add_annotation(
            dict(font=dict(size=30), text=title, xref='x' + str(num) + ' domain', yref='y' + str(num) + ' domain',
                 x=0.5, y=1.15, showarrow=False))

    if num == 1:
        fig.update_layout({'xaxis': {'showline': False, 'showgrid': False, 'visible': False, 'scaleratio': 1,
                                     'scaleanchor': 'x'}})
        fig.update_layout({'yaxis': {'showline': False, 'showgrid': False, 'visible': False, 'scaleratio': 1,
                                     'scaleanchor': 'x'}})

        fig.add_layout_image(dict(source='https://raw.githubusercontent.com/aogando/Farola/master/PlanoMTI.png',
                                  xref='x domain', yref='y domain', y=0.512, x=0.496,
                                  sizex=1.04, sizey=1.04, xanchor='center', yanchor='middle'))

        fig.add_annotation(dict(font=dict(size=25), text=str(name[0:2]) + ':' + str(name[3:]), xref='x domain',
                                yref='y domain', x=-0.09, y=0.5, showarrow=False, textangle=-90))

        fig.add_annotation(
            dict(font=dict(size=30), text=title, xref='x domain', yref='y domain', x=0.5, y=1.15, showarrow=False))

    else:
        fig.update_layout({'xaxis' + str(num): {'showline': False, 'showgrid': False, 'visible': False, 'scaleratio': 1,
                                                'scaleanchor': 'x'}})
        fig.update_layout({'yaxis' + str(num): {'showline': False, 'showgrid': False, 'visible': False, 'scaleratio': 1,
                                                'scaleanchor': 'x'}})

        fig.add_layout_image(dict(source='https://raw.githubusercontent.com/aogando/Farola/master/PlanoMTI.png',
                                  xref='x' + str(num) + ' domain', yref='y' + str(num) + ' domain', y=0.512, x=0.496,
                                  sizex=1.04, sizey=1.04, xanchor='center', yanchor='middle'))
        # sizex=1.11, sizey=1.11, xanchor='center', yanchor='middle'))

        # fig.add_annotation(dict(text=str(name[0:2]) + ':' + str(name[3:]), xref='x' + str(num) + ' domain',
        #                         yref='y' + str(num) + ' domain', x=0, y=1.05, showarrow=False))


#zminT, zmaxT = minmax("Paper/temp")
#zminHR, zmaxHR = minmax("Paper/hr")
#zminCO2, zmaxCO2 = minmax("Paper/co2")
zminT, zmaxT = minmax("temp2")
#zminHR, zmaxHR = minmax("hum")
#zminCO2, zmaxCO2 = minmax("Paper/co2")

# fig = subplots.make_subplots(rows=3, cols=3, vertical_spacing=0.04, horizontal_spacing=0.015)
fig = subplots.make_subplots(rows=3, cols=3, vertical_spacing=0.0, horizontal_spacing=0.015)
layout = go.Layout(autosize=True,
                   # width=1044,
                   # height=300,
                   margin=dict(r=5, l=50, b=150, t=60),
                   plot_bgcolor='rgba(0,0,0,0)',
                   )
fig.update_layout(layout)

#plot(fig, "['temp12.56'].csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=1, col=1, num=1,
#     showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.84)
#plot(fig, "Paper/temp/['temp13.30']2.csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=2, col=1, num=4,
#     showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.84)
#plot(fig, "Paper/temp/['temp14.00']2.csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=3, col=1, num=7,
#     showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.84)

plot(fig, "['temp12.3'].csv", gridsize=1, title='Relative Humidity', legend='[%]', row=1, col=1, num=1,
     showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.505)
plot(fig, "['temp13.0'].csv", gridsize=1, title='Relative Humidity', legend='[%]', row=2, col=1, num=4,
     showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.505)
plot(fig, "['temp13.3'].csv", gridsize=1, title='Relative Humidity', legend='[%]', row=3, col=1, num=7,
     showscale=False, zmin=zminT, zmax=zmaxT, ypos=0.505)
# plot(fig, "Paper/temp/['temp10.30'] (2).csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=1, col=2, num=2, showscale=False, zmin=zminT, zmax=zminT, ypos=0.505)
# plot(fig, "Paper/temp/['temp10.42'] (2).csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=2, col=2, num=5, showscale=False, zmin=zminT, zmax=zminT, ypos=0.505)
# plot(fig, "Paper/temp/['temp10.55'] (2).csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=3, col=2, num=8, showscale=False, zmin=zminT, zmax=zminT, ypos=0.505)
# plot(fig, "Paper/temp/['temp10.30'] (3).csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=1, col=3, num=3, showscale=False, zmin=zminT, zmax=zminT, ypos=0.16)
# plot(fig, "Paper/temp/['temp10.42'] (3).csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=2, col=3, num=6, showscale=False, zmin=zminT, zmax=zminT, ypos=0.16)
# plot(fig, "Paper/temp/['temp10.55'] (3).csv", gridsize=0.5, title='Temperature', legend='[ºC]', row=3, col=3, num=9, showscale=False, zmin=zminT, zmax=zminT, ypos=0.16)

plot(fig, "['hum9.0'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=1, col=2, num=2,
     showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.505)
plot(fig, "['hum9.45'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=2, col=2, num=5,
     showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.505)
plot(fig, "['hum10.3'].csv", gridsize=0.5, title='Relative Humidity', legend='[%]', row=3, col=2, num=8,
     showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.505)

plot(fig, "['hum9.0'].csv", gridsize=0.1, title='Relative Humidity', legend='[%]', row=1, col=3, num=3,
     showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.505)
plot(fig, "['hum9.45'].csv", gridsize=0.1, title='Relative Humidity', legend='[%]', row=2, col=3, num=6,
     showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.505)
plot(fig, "['hum10.3'].csv", gridsize=0.1, title='Relative Humidity', legend='[%]', row=3, col=3, num=9,
     showscale=False, zmin=zminHR, zmax=zmaxHR, ypos=0.505)
#plot(fig, "Paper/co2/['co212.56'].csv", gridsize=0.5, title='CO<sub>2</sub> Concentration', legend='[ppm]', row=1,
#     col=3, num=3, showscale=False, zmin=zminCO2, zmax=zmaxCO2, ypos=0.16)
#plot(fig, "Paper/co2/['co213.30'].csv", gridsize=0.5, title='CO<sub>2</sub> Concentration', legend='[ppm]', row=2,
#     col=3, num=6, showscale=False, zmin=zminCO2, zmax=zmaxCO2, ypos=0.16)
#plot(fig, "Paper/co2/['co214.00'].csv", gridsize=0.5, title='CO<sub>2</sub> Concentration', legend='[ppm]', row=3,
#     col=3, num=9, showscale=False, zmin=zminCO2, zmax=zmaxCO2, ypos=0.16)
# fig_temp = plot("Paper/temps/['temp10.31'].csv", 'Temperature [ºC]')
# fig_hum = plot("Paper/hr/['hum10.29'].csv", 'Relative Humidity [%]')
# fig_hum = plot("Paper/hr/['hum10.30'].csv", 'Relative Humidity [%]')
# fig_hum = plot("Paper/hr/['hum10.31'].csv", 'Relative Humidity [%]')
# fig_co = plot("Paper/co2/['co210.29'].csv", 'CO<sub>2</sub> Concentration [ppm]')
# fig_co = plot("Paper/co2/['co210.30'].csv", 'CO<sub>2</sub> Concentration [ppm]')
# fig_co = plot("Paper/co2/['co210.31'].csv", 'CO<sub>2</sub> Concentration [ppm]')

# fig = subplots.make_subplots(figure=[fig_temp, fig_hum, fig_co], rows=3, cols


# fig.add_annotation(dict(text=str(name[0:2]) + ':' + str(name[3:]), xref='x' + str(num) + ' domain',
#                         yref='y' + str(num) + ' domain', x=0, y=1.05, showarrow=False))

plotly.offline.plot(fig,
                    auto_open=True,
                    image_width=1044,
                    image_height=493, )
