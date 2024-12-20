import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from IPython.core.pylabtools import figsize
from statsmodels.stats import diagnostic
from scipy import stats
import json
import ast
from pathlib import Path
from collections import Counter
import plotly.express as px
from googleapiclient.discovery import build
import time
from scipy.stats import lognorm, shapiro, probplot, kstest, norm, linregress, stats
from scipy.stats import norm, lognorm, expon, gamma, beta, powerlaw, gaussian_kde
import plotly.graph_objects as go
from scipy.stats import linregress, kstest
from sklearn.preprocessing import MinMaxScaler
import plotly.io as pio

from config.ada_config.config import CONFIG

########################################### Actor Age ###########################################
def get_movie_year(movie_year, lower_bound=1906, upper_bound=2025, typo_bound=1025):
    if type(movie_year) == str:
        new_movie_year = movie_year.split("-")[0]
        if int(new_movie_year) > lower_bound and int(new_movie_year) < upper_bound:
            return new_movie_year
        elif int(new_movie_year) < typo_bound and int(new_movie_year) > 1000:
            return str(int(new_movie_year) + 1000)
        else:
            return None
        
    elif type(movie_year) == int:
        if movie_year > lower_bound and movie_year < upper_bound:
            return movie_year
    else:
        return None

def get_movies_with_dates(movie_date):
    if type(movie_date) == str:
        if len(movie_date) > 4:
            return movie_date
        
def get_month(movie_date):
    if type(movie_date) == str:
        if len(movie_date) > 4:
            return movie_date.split("-")[1]
        else:
            return None
    else:
        return None
    

def test_fit_dist_qplot(ages, x_label='Actor Age', y_label='Density', dist="norm", bins=30):
    lower_bound = np.percentile(ages, 2.5)
    upper_bound = np.percentile(ages, 97.5)
    filtered_ages = ages[(ages >= lower_bound) & (ages <= upper_bound)]
    if dist == "norm":
        D, p_value = shapiro(filtered_ages)
        x = np.linspace(min(filtered_ages), max(filtered_ages), 100)
        mu, std = norm.fit(filtered_ages)
        pdf_fitted = norm.pdf(x, mu, std)

    elif dist == "lognorm":
        log_transformed_ages = np.log(filtered_ages)
        shape, loc, scale = lognorm.fit(filtered_ages, floc=0)
        D, p_value = kstest(filtered_ages, 'lognorm', args=(shape, loc, scale))
        x = np.linspace(min(filtered_ages), max(filtered_ages), 100)
        pdf_fitted = lognorm.pdf(x, shape, loc, scale)

    else:
        raise ValueError("Invalid distribution type. Choose 'norm' or 'lognorm'.")

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].hist(filtered_ages, bins=bins, density=True, alpha=0.5, label='Age Distribution Histogram')
    if dist == "norm":
        ax[0].plot(x, pdf_fitted, 'r-', label='Fitted Normal PDF', linewidth=2)
    elif dist == "lognorm":
        ax[0].plot(x, pdf_fitted, 'r-', label='Fitted Log-Normal PDF', linewidth=2)

    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    ax[0].set_title(f'Fitted Log-Normal Distribution to {x_label} Data')
    ax[0].legend()

    probplot(log_transformed_ages, dist="norm", plot=plt)
    ax[1].set_title("Q-Q Plot of Log-Transformed Data")
    ax[1].set_xlabel("Theoretical Quantiles")
    ax[1].set_ylabel("Sample Quantiles")
    ax[1].grid(True)
    plt.show()
    plt.tight_layout()

    print(f"Kolmogorov-Smirnov D statistic: {D:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value > 0.05:
        print("The data follows a log-normal distribution (fail to reject H0 at 5% significance level).")
    else:
        print("The data does not follow a log-normal distribution (reject H0 at 5% significance level).")


###################################################### Crew ######################################################
def preprocess_populairty_crew(df):
    df_popularity_crew = df.copy()

    df_popularity_crew = df_popularity_crew.dropna(subset=["star_1_popularity", "star_2_popularity", "star_3_popularity", "Director_popularity", "Writer_popularity", "Producer_popularity"])
    df_popularity_crew[["star_1_popularity", "star_2_popularity", "star_3_popularity", "star_4_popularity", "star_5_popularity", "Director_popularity", "Writer_popularity", "Producer_popularity"]].describe()

    # Columns to scale
    columns = ["star_1_popularity", "star_2_popularity", "star_3_popularity", "star_4_popularity", "star_5_popularity", "Director_popularity", "Writer_popularity", "Producer_popularity"]

    # Apply Min-Max Scaling
    scaler = MinMaxScaler()
    df_popularity_crew[columns] = scaler.fit_transform(df_popularity_crew[columns])

    # Calculate total popularity
    df_popularity_crew.loc[:,'total_popularity'] = df_popularity_crew[columns].sum(axis=1)
    df_popularity_crew.loc[:,'cast_popularity'] = df_popularity_crew[["star_1_popularity", "star_2_popularity", "star_3_popularity", "star_4_popularity", "star_5_popularity"]].sum(axis=1)
    
    return df_popularity_crew

def plot_popularity_crew(df_popularity_crew_remakes, df_popularity_crew_originals, df_popularity_crew_rest, column='total_popularity'):
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    sns.kdeplot(df_popularity_crew_remakes[column], label=f'Remakes {column.replace("_", " ")}', color='blue', fill=True, ax=ax)
    ax.vlines(df_popularity_crew_remakes[column].mean(), 0, 2, color='blue', linestyle='--', label=f'Mean Remakes {column.replace("_", " ")}')
    sns.kdeplot(df_popularity_crew_originals[column], label=f'Originals {column.replace("_", " ")}', color='red', fill=True, ax=ax)
    ax.vlines(df_popularity_crew_originals[column].mean(), 0, 2, color='red', linestyle='--', label=f'Mean Originals {column.replace("_", " ")}')
    sns.kdeplot(df_popularity_crew_rest[column], label=f'Rest {column.replace("_", " ")}', color='green', fill=True, ax=ax)
    ax.vlines(df_popularity_crew_rest[column].mean(), 0, 2, color='green', linestyle='--', label=f'Mean Rest {column.replace("_", " ")}')
    # sns.kdeplot(df_popularity_crew['total_popularity'], label='Whole Dataset', color='purple', fill=True, ax=ax)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_popularity_crew_plotly_smooth(df_popularity_crew_remakes, df_popularity_crew_originals, df_popularity_crew_rest, column='total_popularity', output_file="popularity_crew_plot.html"):
    """
    Plot interactive smoothed KDE-like plot of total popularity for remakes, originals, and rest datasets using Plotly.

    Parameters:
        df_popularity_crew_remakes (pd.DataFrame): DataFrame containing popularity data for remakes.
        df_popularity_crew_originals (pd.DataFrame): DataFrame containing popularity data for originals.
        df_popularity_crew_rest (pd.DataFrame): DataFrame containing popularity data for rest of the movies.
        column (str): The column to visualize (default is 'total_popularity').
        output_file (str): Output HTML file for saving the plot.
    """
    # Define color scheme
    colors = {
        'whole_dataset': '#636EFA',  # Blue
        'remakes': '#EF553B',        # Red
        'originals': '#00CC96',      # Green
        'rest': '#AB63FA'           # Purple
    }

    # Generate KDE data
    def kde_smooth_data(series, num_points=500):
        data = series.dropna()
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), num_points)
        y_vals = kde(x_vals)
        return x_vals, y_vals

    # Generate smoothed data for each dataset
    x_remakes, y_remakes = kde_smooth_data(df_popularity_crew_remakes[column])
    x_originals, y_originals = kde_smooth_data(df_popularity_crew_originals[column])
    x_rest, y_rest = kde_smooth_data(df_popularity_crew_rest[column])

    # Create traces for each dataset
    fig = go.Figure()

    # Remakes
    fig.add_trace(go.Scatter(
        x=x_remakes, y=y_remakes,
        mode='lines',
        name=f'Remakes {column.replace("_", " ")}',
        line=dict(color=colors['remakes']),
        fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=[df_popularity_crew_remakes[column].mean()], y=[0],
        mode='markers',
        name=f'Mean Remakes {column.replace("_", " ")}',
        marker=dict(color=colors['remakes'], symbol='x', size=10)
    ))

    # Originals
    fig.add_trace(go.Scatter(
        x=x_originals, y=y_originals,
        mode='lines',
        name=f'Originals {column.replace("_", " ")}',
        line=dict(color=colors['originals']),
        fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=[df_popularity_crew_originals[column].mean()], y=[0],
        mode='markers',
        name=f'Mean Originals {column.replace("_", " ")}',
        marker=dict(color=colors['originals'], symbol='x', size=10)
    ))

    # Rest
    fig.add_trace(go.Scatter(
        x=x_rest, y=y_rest,
        mode='lines',
        name=f'Rest {column.replace("_", " ")}',
        line=dict(color=colors['rest']),
        fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=[df_popularity_crew_rest[column].mean()], y=[0],
        mode='markers',
        name=f'Mean Rest {column.replace("_", " ")}',
        marker=dict(color=colors['rest'], symbol='x', size=10)
    ))

    # Update layout
    fig.update_layout(
        title=f"Popularity Analysis of {column.replace('_', ' ')}",
        xaxis_title=column.replace("_", " "),
        yaxis_title="Density",
        template="plotly_white",
        autosize=True,
        height=600,
        width=900,
        legend_title="Legend"
    )

    # Save and display the plot
    fig.write_html(output_file, include_plotlyjs="cdn", auto_open=True)
    print(f"Plot saved to {output_file}")
    fig.show()


def plot_people_perception_interactive(df_movies, df_remakes, df_originals, df_rest, column_name, colors, bins=None, is_log=False, output_file="people_perception.html"):
    """
    Create an interactive plot to visualize scaled counts of a column across multiple datasets.

    Parameters:
        df_movies (pd.DataFrame): The dataset containing all movies.
        df_remakes (pd.DataFrame): The dataset containing remakes.
        df_originals (pd.DataFrame): The dataset containing originals.
        df_rest (pd.DataFrame): The dataset containing the rest of the movies.
        column_name (str): The column to analyze.
        bins (int or list, optional): Number of bins or bin edges for the histograms.
        is_log (bool, optional): Whether to use a log scale for the x-axis.
        output_file (str, optional): The filename to save the interactive plot as an HTML file.

    Returns:
        None. Displays the plot and saves it to an HTML file.
    """
    # Compute histograms
    counts_whole, edges_whole = np.histogram(df_movies[column_name].dropna(), bins=bins)
    counts_remakes, edges_remakes = np.histogram(df_remakes[column_name].dropna(), bins=bins)
    counts_originals, edges_originals = np.histogram(df_originals[column_name].dropna(), bins=bins)
    counts_rest, edges_rest = np.histogram(df_rest[column_name].dropna(), bins=bins)

    # Compute bin centers
    bin_centers_whole = (edges_whole[:-1] + edges_whole[1:]) / 2
    bin_centers_remakes = (edges_remakes[:-1] + edges_remakes[1:]) / 2
    bin_centers_originals = (edges_originals[:-1] + edges_originals[1:]) / 2
    bin_centers_rest = (edges_rest[:-1] + edges_rest[1:]) / 2

    # Filter zero-count bins for plotting and scaling
    def filter_nonzero(counts, bin_centers):
        mask = counts > 0
        return counts[mask], bin_centers[mask]

    counts_whole, bin_centers_whole = filter_nonzero(counts_whole, bin_centers_whole)
    counts_remakes, bin_centers_remakes = filter_nonzero(counts_remakes, bin_centers_remakes)
    counts_originals, bin_centers_originals = filter_nonzero(counts_originals, bin_centers_originals)
    counts_rest, bin_centers_rest = filter_nonzero(counts_rest, bin_centers_rest)

    # Min-Max Scaling
    def min_max_scale(arr):
        return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() != arr.min() else arr

    scaled_counts_whole = min_max_scale(counts_whole)
    scaled_counts_remakes = min_max_scale(counts_remakes)
    scaled_counts_originals = min_max_scale(counts_originals)
    scaled_counts_rest = min_max_scale(counts_rest)

    # Define colors for each dataset
    # colors = {
    #     "whole_dataset": "#636EFA",  # Blue
    #     "remakes": "#EF553B",        # Red
    #     "originals": "#00CC96",      # Green
    #     "rest": "#AB63FA"           # Purple
    # }

    # Create traces for each dataset
    traces = [
        go.Scatter(
            x=bin_centers_whole,
            y=scaled_counts_whole,
            mode="lines+markers",
            name="Scaled Whole Dataset",
            line=dict(color=colors["whole_dataset"]),
            marker=dict(symbol="circle", size=6)
        ),
        go.Scatter(
            x=bin_centers_remakes,
            y=scaled_counts_remakes,
            mode="lines+markers",
            name="Scaled Remakes",
            line=dict(color=colors["remakes"]),
            marker=dict(symbol="square", size=6)
        ),
        go.Scatter(
            x=bin_centers_originals,
            y=scaled_counts_originals,
            mode="lines+markers",
            name="Scaled Originals",
            line=dict(color=colors["originals"]),
            marker=dict(symbol="triangle-up", size=6)
        ),
        go.Scatter(
            x=bin_centers_rest,
            y=scaled_counts_rest,
            mode="lines+markers",
            name="Scaled Rest",
            line=dict(color=colors["rest"]),
            marker=dict(symbol="diamond", size=6)
        )
    ]

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title=f"Log of Min-Max Scaled Counts of Different {column_name.replace('_', ' ')} Scores",
        xaxis=dict(
            title=f"{column_name.replace('_', ' ')} Scores (Log Scale)" if is_log else f"{column_name.replace('_', ' ')} Scores",
            type="log" if is_log else "linear"
        ),
        yaxis=dict(title="Scaled Counts (0-1)"),
        template="plotly_white",
        legend=dict(title="Dataset", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        autosize=True,
        height=600,
        width=800
    )

    # Save to HTML
    pio.write_html(fig, file=output_file, auto_open=True, include_plotlyjs="cdn")
    print(f"Interactive plot saved to {output_file}")

    # Show the plot
    fig.show()