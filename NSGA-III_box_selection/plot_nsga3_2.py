import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import itertools
from pymoo.factory import get_decision_making
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)




#resPath = '/media/ana/Datos/R-tableCalibrations/Optimization/Pruebas_nsgaIII/lumSimulation/Results/20210521_12:49'
#pareto_df = pd.read_csv('result_obj2.csv', sep=';', decimal=',')
pareto_df = pd.read_csv('result_obj.csv', sep='?')
pareto_df2 = pareto_df.copy()
results_df = pd.read_csv('result_X_.csv', sep='?')
boxes= results_df.iloc[:,11-8:11]
boxes_dist = boxes.sum()/len(boxes)
b_s = boxes.sum(axis=1)
#g8 = np.where(b_s==8)[0]
#g7 = np.where(b_s==7)[0]
#g6 = np.where(b_s==6)[0]
#g5 = np.where(b_s==5)[0]
#g4 = np.where(b_s==4)[0]
#g3 = np.where(b_s==3)[0]
r1 =np.array(pareto_df.iloc[:,0]).reshape(-1,1)
r1_scalar = MinMaxScaler(feature_range=(0, 1))
r1_scalar.fit(r1)
r1_scaled=r1_scalar.transform(r1)
pareto_df.iloc[:,0]=r1_scaled

r2 =np.array(pareto_df.iloc[:,1]).reshape(-1,1)
r2_scalar = MinMaxScaler(feature_range=(0, 1))
r2_scalar.fit(r2)
r2_scaled=r2_scalar.transform(r2)
pareto_df.iloc[:,1]=r2_scaled

r3 =np.array(pareto_df.iloc[:,2]).reshape(-1,1)
r3_scalar = MinMaxScaler(feature_range=(0, 1))
r3_scalar.fit(r3)
r3_scaled=r3_scalar.transform(r3)
pareto_df.iloc[:,2]=r3_scaled

#r4 =np.array(pareto_df.iloc[:,3]).reshape(-1,1)
#r4_scalar = MinMaxScaler(feature_range=(0, 1))
#r4_scalar.fit(r4)
#r4_scaled=r4_scalar.transform(r4)
#pareto_df.iloc[:,3]=r4_scaled

#pareto_df2.loc[pareto_df2.shape[0]]=pd.DataFrame(np.array([3,4,5,2]).reshape(1,-1))
#pareto_df2= pareto_df2.append(pd.DataFrame(np.array([3,4,5,2]).reshape(1,-1), columns=pareto_df2.columns), ignore_index=True)


F_var = ['Temperatures', 'Relative humidity', 'CO2', 'Number of boxes']
pareto_df['Number of boxes'] = pd.Series(np.array(pareto_df['Number of boxes'], dtype=int)).astype('category')
resultados = pd.concat([results_df, pareto_df.iloc[:,3]], axis=1)




pareto_df = pareto_df.sort_values(by=['Number of boxes'])
pareto_df=pareto_df.reset_index(drop=True)

pareto_df2 = pareto_df2.sort_values(by=['Number of boxes'])
pareto_df2=pareto_df2.reset_index(drop=True)

resultados = resultados.sort_values(by=['Number of boxes'])
resultados=resultados.reset_index(drop=True)




weights = np.array([0.3, 0.3, 0.3, 0.1])
opt_index, pseudo_weights = get_decision_making("pseudo-weights", weights).do(
    pareto_df2[F_var], return_pseudo_weights=True)

#data = np.loadtxt('temp_X.txt', delimiter=',', skiprows=1, dtype=str)
import plotly.express as px

fig = make_subplots(rows=1,
                    cols=2,
                    specs=[[{'type': 'scatter3d'}, {'type': 'bar'}]],
                    horizontal_spacing=0.02, column_widths=[0.6, 0.4])


#fig.add_trace(px.scatter_3d(pareto_df, x='Temperatures', y='Relative humidity', z='CO2',
#             color='Number of boxes',color_discrete_sequence=['dimgrey','black','darkgreen','blue']), row=1, col=1)
#fig.add_traces([go.Scatter3d(x=pareto_df['Temperatures'], y=pareto_df['Relative humidity'], z=pareto_df['CO2'],
#             marker=dict(size=10, showscale=False,color=pareto_df['Number of boxes']),
#                             mode='markers')], rows=1, cols=2)

#3
fig.add_trace(go.Scatter3d(x=pareto_df['Temperatures'][0:7], y=pareto_df['Relative humidity'][0:7], z=pareto_df['CO2'][0:7],
             marker=dict(size=11, showscale=False,color='blue'),
                             mode='markers', name='3'))
#4
fig.add_trace(go.Scatter3d(x=pareto_df['Temperatures'][7:11], y=pareto_df['Relative humidity'][7:11], z=pareto_df['CO2'][7:11],
             marker=dict(size=11, showscale=False,color='lime'),
                             mode='markers', name='4'))
#5#
#fig.add_trace(go.Scatter3d(x=pareto_df['Temperatures'][11:17], y=pareto_df['Relative humidity'][11:17], z=pareto_df['CO2'][11:17],
#             marker=dict(size=11, showscale=False,color='blue'),
#                             mode='markers', name='5'))
#6
#fig.add_trace(go.Scatter3d(x=np.array(pareto_df['Temperatures'][10]), y=np.array(pareto_df['Relative humidity'][10]), z=np.array(pareto_df['CO2'][10]),
#             marker=dict(size=11, showscale=False,color='lime'),
#                             mode='markers', name='6'))
#

fig.add_trace(go.Scatter3d(x=[pareto_df.loc[opt_index, 'Temperatures']],
                             y=[pareto_df.loc[opt_index, 'Relative humidity']],
                             z=[pareto_df.loc[opt_index, 'CO2']],
                            mode='markers',
                             #marker=dict(color='rgba(0,0,0,0)',size=15,line=dict(color='red', width=24), opacity='.6'),
                             marker=dict(color='red',size=23,line=dict(color='black', width=50), opacity=0.5),
                             text=['Temperatures','Relative humidity','CO2'],
                             showlegend=False))
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')




bb = ['D1', 'D2','D3','D4','D5','D6','D7','D8']
fig.add_trace(
    go.Bar(x=bb,y=boxes_dist, showlegend=False,
           marker=dict(color='azure',showscale=False),
           marker_line=dict(width=[4,4,4,4,4,4,4,4], color=['blue','blue','red','red','blue','red','blue','blue'])),
    row=1, col=2
)
#fig.add_trace(
#    go.Bar(x=np.array(bb)[np.array([0,2,5])],y=boxes_dist[[0,2,5]], showlegend=False,
#           marker=dict(color='blue',showscale=False),
#           marker_line=dict(width=1.5, color='red')),
#    row=1, col=2
#)
fig.update_yaxes(tickfont=dict(size=23), title_text='Relative frequency', title_font = {'size':26}, domain=[0.03,0.92])
fig.update_xaxes(tickfont=dict(size=24), title_text='Devices',title_font = {'size':26})
#fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
fig.update_yaxes(showgrid=True,gridcolor='black', gridwidth=0.5, range=[0,1.01])
#fig.update_xaxes(title_text='Temperature',title_standoff = 70, row=1, col=2)
#fig.update_layout({'scene' + str(1): {'bgcolor': 'white'}})

fig.update_layout({'scene': {
                                              'xaxis': {'title': {'text': F_var[0], 'font':dict(size=25)}, 'linecolor': 'white','tickfont' : dict(size=16),'tick0':0},
                                              'yaxis': {'title': {'text': F_var[1],'font':dict(size=25)}, 'linecolor': 'white',  'tickfont' : dict(size=16),'tick0':0 },
                                              'zaxis': {'title': {'text': 'CO{}'.format(get_sub('2')),'font':dict(size=25)}, 'linecolor': 'white','tickfont' : dict(size=16),'tick0':0}}},
    legend=dict(title_text='Number of boxes',title_font = {'size': 28, 'color':'black'},
                yanchor="top",
                y=1.02,
                xanchor="left",
                x=0,
traceorder="reversed",
        font=dict(
            size=29,
            color="black"
        ),
        bgcolor='rgba(0,0,0,0)',
    ))

fig.update_layout({"autosize": True, 'width': 2000, 'height': 1000})
#fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
fig.update_scenes(yaxis_showgrid=True, yaxis_gridwidth=1, yaxis_gridcolor='black', yaxis_zeroline=True, yaxis_zerolinecolor='black')
fig.update_scenes(xaxis_showgrid=True, xaxis_gridwidth=1, xaxis_gridcolor='black',xaxis_zeroline=True,xaxis_zerolinecolor='black')
fig.update_scenes(zaxis_showgrid=True, zaxis_gridwidth=1, zaxis_gridcolor='black',zaxis_zeroline=True,zaxis_zerolinecolor='black')

plotly.offline.plot(fig, image='svg', filename='NSGA-3.html')

#, annotations=[dict(text='Number of boxes', showarrow=False,y=0.93, x=0.96,xref='paper',
#           yref='paper' ,font=dict(
#           size=24,
#           color="black"
#       ))]

#sns.set_style("whitegrid")
#fig=plt.figure()
#sns.barplot(x=bb, y=boxes_dist,  palette="Blues_d")
#plt.ylabel('Relative Frequency')
#plt.xlabel('Boxes')
#plt.ylim(0,1)
