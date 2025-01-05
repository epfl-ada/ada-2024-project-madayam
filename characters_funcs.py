import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.pylabtools import figsize
from statsmodels.stats import diagnostic
from scipy import stats
from scipy.stats import lognorm, shapiro, probplot, kstest, norm, linregress, stats
from scipy.stats import norm, lognorm, expon, gamma, beta, powerlaw, gaussian_kde
import plotly.graph_objects as go
from scipy.stats import linregress, kstest
from sklearn.preprocessing import MinMaxScaler
import plotly.io as pio

from config.ada_config.config import CONFIG

########################################### Actor Age ###########################################
def get_movie_year(movie_year, lower_bound=1906, upper_bound=2025, typo_bound=1025):
    """
    Extract the movie year from the movie_year column.

    Parameters:
        movie_year (str or int): The movie year to extract.
        lower_bound (int): The lower bound for the movie year (default is 1906).
        upper_bound (int): The upper bound for the movie year (default is 2025).
        typo_bound (int): The typo bound for the movie year (default is 1025).
    
    Returns:
        str or int: The extracted movie year if it is within the bounds, otherwise None.
    """
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

    if dist == "norm":
        probplot(filtered_ages, dist="norm", plot=ax[1])
        ax[1].set_title("Q-Q Plot of Normal Distribution")
    elif dist == "lognorm":
        probplot(log_transformed_ages, dist="norm", plot=ax[1])
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
    """
    Preprocess the popularity data for the crew members in the dataset.

    Parameters:
        df (pd.DataFrame): DataFrame containing the movie data.
    
    Returns:
        pd.DataFrame: DataFrame with the crew popularity data preprocessed.
    """
    df_popularity_crew = df.copy()

    df_popularity_crew = df_popularity_crew.dropna(subset=["star_1_popularity", "star_2_popularity", "star_3_popularity", "Director_popularity", "Writer_popularity", "Producer_popularity"])
    df_popularity_crew[["star_1_popularity", "star_2_popularity", "star_3_popularity", "star_4_popularity", "star_5_popularity", "Director_popularity", "Writer_popularity", "Producer_popularity"]].describe()

    columns = ["star_1_popularity", "star_2_popularity", "star_3_popularity", "star_4_popularity", "star_5_popularity", "Director_popularity", "Writer_popularity", "Producer_popularity"]

    scaler = MinMaxScaler()
    df_popularity_crew[columns] = scaler.fit_transform(df_popularity_crew[columns])

    df_popularity_crew.loc[:,'total_popularity'] = df_popularity_crew[columns].sum(axis=1)
    df_popularity_crew.loc[:,'cast_popularity'] = df_popularity_crew[["star_1_popularity", "star_2_popularity", "star_3_popularity", "star_4_popularity", "star_5_popularity"]].sum(axis=1)
    
    return df_popularity_crew

def plot_popularity_crew(df_popularity_crew_remakes, df_popularity_crew_originals, df_popularity_crew_rest, column='total_popularity'):
    """
    Plot KDE plots for total popularity for remakes, originals, and rest datasets.

    Parameters:
        df_popularity_crew_remakes (pd.DataFrame): DataFrame containing popularity data for remakes.
        df_popularity_crew_originals (pd.DataFrame): DataFrame containing popularity data for originals.
        df_popularity_crew_rest (pd.DataFrame): DataFrame containing popularity data for rest of the movies.
        column (str): The column to visualize (default is 'total_popularity').
    """
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    sns.kdeplot(df_popularity_crew_remakes[column], label=f'Remakes {column.replace("_", " ")}', color='blue', fill=True, ax=ax)
    ax.vlines(df_popularity_crew_remakes[column].mean(), 0, 2, color='blue', linestyle='--', label=f'Mean Remakes {column.replace("_", " ")}')
    sns.kdeplot(df_popularity_crew_originals[column], label=f'Originals {column.replace("_", " ")}', color='red', fill=True, ax=ax)
    ax.vlines(df_popularity_crew_originals[column].mean(), 0, 2, color='red', linestyle='--', label=f'Mean Originals {column.replace("_", " ")}')
    sns.kdeplot(df_popularity_crew_rest[column], label=f'Rest {column.replace("_", " ")}', color='green', fill=True, ax=ax)
    ax.vlines(df_popularity_crew_rest[column].mean(), 0, 2, color='green', linestyle='--', label=f'Mean Rest {column.replace("_", " ")}')
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
    colors = {
        'whole_dataset': '#636EFA',
        'remakes': '#EF553B',
        'originals': '#00CC96',
        'rest': '#AB63FA'
    }

    def kde_smooth_data(series, num_points=500):
        data = series.dropna()
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), num_points)
        y_vals = kde(x_vals)
        return x_vals, y_vals

    x_remakes, y_remakes = kde_smooth_data(df_popularity_crew_remakes[column])
    x_originals, y_originals = kde_smooth_data(df_popularity_crew_originals[column])
    x_rest, y_rest = kde_smooth_data(df_popularity_crew_rest[column])

    fig = go.Figure()

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

    fig.write_html(output_file, include_plotlyjs="cdn", auto_open=True)
    print(f"Plot saved to {output_file}")
    fig.show()

def plot_average_star_popularity(df_movies, df_remakes, df_originals, colors, output_file="average_star_popularity.html"):
    """
    Create interactive bar plots for the average star popularity across the top 3 stars 
    for all movies, remakes, and originals datasets.

    Parameters:
        df_movies (pd.DataFrame): The dataset containing all movies.
        df_remakes (pd.DataFrame): The dataset containing remakes.
        df_originals (pd.DataFrame): The dataset containing originals.
        colors (dict): Dictionary of colors for each dataset.
        output_file (str): The filename to save the interactive plots as an HTML file.

    Returns:
        None. Saves the plots as an HTML file and displays them.
    """
    average_movies = (
        df_movies[["star_1_popularity", "star_2_popularity", "star_3_popularity"]].mean().mean()
    )
    average_remakes = (
        df_remakes[["star_1_popularity", "star_2_popularity", "star_3_popularity"]].mean().mean()
    )
    average_originals = (
        df_originals[["star_1_popularity", "star_2_popularity", "star_3_popularity"]].mean().mean()
    )

    datasets = ["All Movies", "Remakes", "Originals"]
    averages = [average_movies, average_remakes, average_originals]

    fig = go.Figure()

    for dataset, average, color in zip(datasets, averages, colors.values()):
        fig.add_trace(go.Bar(
            x=[dataset],
            y=[average],
            name=dataset,
            marker=dict(color=color),
            text=[f"{average:.2f}"],
            textposition="outside"
        ))

    fig.update_layout(
        title="Average Top-3 Star Popularity Across Datasets",
        xaxis=dict(title="Dataset"),
        yaxis=dict(title="Average Popularity"),
        barmode="group",
        legend=dict(title="Dataset"),
        template="plotly_white",
        autosize=True,
        height=600,
        width=800
    )

    fig.write_html(output_file, auto_open=True, include_plotlyjs="cdn")
    print(f"Interactive average star popularity plot saved to {output_file}")

    fig.show()