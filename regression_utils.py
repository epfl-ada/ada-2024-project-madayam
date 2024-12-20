
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

import random
import pandas as pd
from collections import deque
from sklearn.metrics import f1_score

import networkx as nx
np.bool = np.bool_
import plotly.io as pio

random.seed(42)
np.random.seed(42)



def plot_significance_plotly_regression_all(model, path_to_save, title='Coefficients and Confidence Intervals'):
    coefficients = model.params
    conf_intervals = model.conf_int()
    conf_intervals.columns = ['lower', 'upper']

    coef_df = pd.concat([coefficients, conf_intervals], axis=1)
    coef_df.columns = ['coef', 'lower', 'upper']
    coef_df = coef_df[coef_df.index != 'const']

    coef_df['significant'] = (coef_df['lower'] > 0) | (coef_df['upper'] < 0)

    fig = go.Figure()

    for i, row in coef_df.iterrows():
        color = 'red' if row['significant'] else 'black'
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[row['coef']],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f'{i} Coef'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[i, i],
                y=[row['lower'], row['upper']],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            )
        )

    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(coef_df) - 0.5,
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Predictor Variables',
        yaxis_title='Coefficient Value',
        xaxis=dict(tickvals=list(range(len(coef_df))), ticktext=coef_df.index, tickangle=45),
        showlegend=False,
        template="plotly_white",
        autosize=True,
        height=600,
        width=800,
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.1,
        y=1.05,
        text="<br><span style='color:red'>Red = Significant</span><br>Black = Not Significant",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=5,
        bgcolor="white"
    )

    fig.show()

    pio.write_html(fig, path_to_save, auto_open=True, include_plotlyjs="cdn")

    return coef_df



def regress(df, has_remake=True):

    if has_remake:
        col_keep = 'has_remake'
        col_remove = 'is_remake'
    else:
        col_keep = 'is_remake'
        col_remove = 'has_remake'

    df = df[df[col_remove] == 0]

    X = df.drop(columns=['has_remake', 'is_remake'])
    y = df[col_keep]
    print('number of samples with positive y:', y.sum())

    X = X.astype(float)
    y = y.astype(float)
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, X).fit()

    print('acc:', (model.predict(X) > 0.5).eq(y).mean())
    print('f1:', f1_score(y, model.predict(X) > 0.5))
    print(model.summary())
    return model



def plot_ate_or_histograms(results, output_file="ate_or_histograms.html", has_remake=True):

    columns = list(results.keys())
    try:
        ate_values = [results[col]['ATE'] for col in columns]
        or_values = [results[col]['OR'] for col in columns]
    except KeyError as e:
        raise ValueError(f"Missing expected key in results: {e}")

    if not ate_values or not or_values:
        raise ValueError("ATE or OR values are empty. Check the input results dictionary.")

    fig_ate = go.Figure()
    if has_remake:
        pre_ = 'Having Remake'
    else:
        pre_ = 'Being Remake'

    if ate_values:
        fig_ate.add_trace(go.Bar(
            x=columns,
            y=ate_values,
            name="ATE",
            marker=dict(color="#636EFA"),
            textposition="outside",
            showlegend=False,
        ))
        ate_std_err = [results[col]['ATE_std_ere'] * 1.96 for col in columns]
        fig_ate.add_trace(go.Scatter(
            x=columns,
            y=ate_values,
            mode='markers',
            marker=dict(size=10, color='red'),
            name="ATE 95% CI",
            error_y=dict(
                type='data',
                array=ate_std_err,
                visible=True
            ),
            showlegend=False,
        ))

    fig_ate.add_shape(
        type="line",
        x0=-0.5,
        x1=len(columns) - 0.5,
        y0=0.05,
        y1=0.05,
        line=dict(color="black", width=1, dash="dash")
    )

    fig_ate.add_shape(
        type="line",
        x0=-0.5,
        x1=len(columns) - 0.5,
        y0=-0.05,
        y1=-0.05,
        line=dict(color="black", width=1, dash="dash")
    )

    fig_ate.update_layout(
        title=pre_ + " ATE Metrics Across Columns",
        xaxis=dict(title="Columns", tickmode="array", tickvals=list(range(len(columns))), ticktext=columns),
        yaxis=dict(title="ATE Values"),
        template="plotly_white",
        autosize=True,
        height=600,
        width=800
    )

    fig_ate.show()
    pio.write_html(fig_ate, output_file.replace(".html", "_ate.html"), auto_open=True, include_plotlyjs="cdn", auto_play=False)

    fig_or = go.Figure()

    if or_values:
        fig_or.add_trace(go.Bar(
            x=columns,
            y=or_values,
            name="OR",
            marker=dict(color="#EF553B"),
            textposition="outside"
        ))

    fig_or.update_layout(
        title=pre_ + " OR Metrics Across Columns",
        xaxis=dict(title="Columns", tickmode="array", tickvals=list(range(len(columns))), ticktext=columns),
        yaxis=dict(title="OR Values"),
        template="plotly_white",
        autosize=True,
        height=600,
        width=800
    )

    fig_or.show()
    pio.write_html(fig_or, output_file.replace(".html", "_or.html"), auto_open=True, include_plotlyjs="cdn", auto_play=False)



def check_each_col_treat(df, has_remake=True):
    if has_remake:
        col_keep = 'has_remake'
        col_remove = 'is_remake'
    else:
        col_keep = 'is_remake'
        col_remove = 'has_remake'

    df = df[df[col_remove] == 0].reset_index(drop=True)

    X = df.drop(columns=['has_remake', 'is_remake'])
    y = df[col_keep].astype('bool')

    result = {}
    print('number of samples with positive y:', y.sum())
    for treat_col in X.columns:
        result[treat_col] = {}
        print()
        print('##############', treat_col, '##############')
        MAX_MATCHING_THRESHOLD = y.std() / y.shape[0] ** 0.5
        if len(X[treat_col].value_counts()) == 2:
            thr = 0.5
        elif 'sentiment' in treat_col:
            thr = 0.0
        else:
            thr = X[treat_col].mean() + X[treat_col].std() * 2
        treatment = y
        covariates = X.drop(columns=treat_col)
        model = sm.Logit(treatment, sm.add_constant(covariates, has_constant='add')).fit()
        print('thr:', thr, 'MAX_MATCHING_THRESHOLD:', MAX_MATCHING_THRESHOLD)
        outcome = df[treat_col] > thr
        df[f'{treat_col}_propensity_score'] = model.predict(sm.add_constant(covariates, has_constant='add'))

        df[f'{treat_col}_outcome'] = outcome
        control_df = df[~treatment]
        treatment_df = df[treatment]
        G = nx.Graph()
        sorted_control_df = control_df.sort_values(by=f'{treat_col}_propensity_score', ascending=True).reset_index(drop=True)
        sorted_treatment_df = treatment_df.sort_values(by=f'{treat_col}_propensity_score', ascending=True).reset_index(drop=True)
        start_treatment_index = 0
        end_treatment_index = 0

        ind_dq = deque()
        score_dq = deque()
        edges = []
        for i, row in sorted_control_df.iterrows():
            while end_treatment_index < len(sorted_treatment_df) and abs(row[f'{treat_col}_propensity_score'] - sorted_treatment_df[f'{treat_col}_propensity_score'].iloc[end_treatment_index]) < MAX_MATCHING_THRESHOLD:
                ind_dq.append(sorted_treatment_df.index[end_treatment_index])
                score_dq.append(sorted_treatment_df[f'{treat_col}_propensity_score'].iloc[end_treatment_index])
                end_treatment_index += 1
            while start_treatment_index < end_treatment_index and abs(row[f'{treat_col}_propensity_score'] - sorted_treatment_df[f'{treat_col}_propensity_score'].iloc[start_treatment_index]) >= MAX_MATCHING_THRESHOLD:
                ind_dq.popleft()
                score_dq.popleft()
                start_treatment_index += 1
            i_score = row[f'{treat_col}_propensity_score']
            sen_thr = 5
            for j, score in zip(ind_dq, score_dq):
                if 1 / sen_thr <= (i_score / (1 - i_score)) / (score / (1 - score)) <= sen_thr:
                    edges.append((j, i + len(sorted_treatment_df)))
        random.seed(42)
        random.shuffle(edges)
        G.add_edges_from(edges)
        nodes = list(sorted_treatment_df.index)
        random.seed(42)
        random.shuffle(nodes)
        G.add_nodes_from(nodes, bipartite=0)
        nodes = [ind_ + len(sorted_treatment_df) for ind_ in range(len(sorted_control_df))]
        random.seed(42)
        random.shuffle(nodes)
        G.add_nodes_from(nodes, bipartite=1)

        matching = nx.bipartite.maximum_matching(G, top_nodes=list(sorted_treatment_df.index))
        print(f'{treat_col} matched:', len(matching) // 2)
        pairs = [[matching[i] - len(sorted_treatment_df), i] for i in range(len(sorted_treatment_df)) if i in matching]

        pairs = np.array(pairs)
        y_control = sorted_control_df.loc[pairs[:, 0], f'{treat_col}_outcome'].values
        y_treatment = sorted_treatment_df.loc[pairs[:, 1], f'{treat_col}_outcome'].values
        ATE_arr = np.array(y_treatment, dtype=float) - np.array(y_control, dtype=float)
        treat_socre = sorted_treatment_df.loc[pairs[:, 1], f'{treat_col}_propensity_score'].values
        control_score = sorted_control_df.loc[pairs[:, 0], f'{treat_col}_propensity_score'].values
        OR_arr = treat_socre / (1 - treat_socre) / (control_score / (1 - control_score))
        print(f'{treat_col} matching ATE:', ATE_arr.mean())
        OR = np.maximum(1 / OR_arr.min(), OR_arr.max())
        print(f'{treat_col} matching OR:', OR)
        result[treat_col]['ATE'] = ATE_arr.mean()
        result[treat_col]['OR'] = OR
        result[treat_col]['ATE_std_ere'] = ATE_arr.std() / (len(ATE_arr) ** 0.5)
    return result