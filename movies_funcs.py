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
from scipy.stats import norm, lognorm, expon, gamma, beta, powerlaw
import plotly.graph_objects as go
from scipy.stats import linregress, kstest
from sklearn.preprocessing import MinMaxScaler
import plotly.io as pio


from config.ada_config.config import CONFIG

###################################################### General Functions ###########################################################
def plot_boxplot(df, column_name, title, ax):
    movie_years_only = df[column_name].dropna().astype(int)
    ax.boxplot(movie_years_only, vert=False)
    ax.set_yticks([])
    ax.set_title(title)
    
    stats = movie_years_only.aggregate(["mean", "median", "std"])
    print(f"Statistics for {title}:")
    print(stats)
    print("="*50)

def plot_boxplot_with_upperbound(df, column_name, title, upper_bound, ax):
    filtered_data = df[df[column_name] < upper_bound][column_name].dropna()

    ax.boxplot(filtered_data, vert=False)
    ax.set_yticks([])
    ax.set_title(title + f" (UB={upper_bound})")

    half_ub_filtered = df[df[column_name] < upper_bound][column_name].dropna()
    first_quartile = half_ub_filtered.quantile(0.25)
    third_quartile = half_ub_filtered.quantile(0.75) 
    stats = half_ub_filtered.aggregate(["mean", "median", "std", "min", "max"])

    print(f"Statistics for {title} (UB={upper_bound}):")
    print(f"First quartile (UB/2): {first_quartile}")
    print(f"Third quartile (UB/2): {third_quartile}")
    print(stats)
    print("="*50)

def cleaning_genres(genres):
    """
    Clean the genres by removing "film" from the genre name.
    """
    if isinstance(genres, str):
        genres = genres.split(", ")
        genres = [genre.lower().replace(" film", "") for genre in genres]
        genres = [genre.title() for genre in genres]
        return ", ".join(genres)

###################################################### Budget and Revenue ###########################################################
def fit_and_plot_distributions(data, column, title, distributions, with_log_transform=True, epsilon=1e-10):
    """
    Fit data to one or multiple distributions, calculate log-likelihood, AIC, and plot results.

    Parameters:
    - data (DataFrame): Input dataset.
    - column (str): Column to fit and plot.
    - title (str): Title for the plot.
    - distributions (list or single): Distribution(s) to fit the data. Accepts single or list of scipy.stats distributions.
    """
    if not isinstance(distributions, list):
        distributions = [distributions]
    
    if with_log_transform:
        # Filter and log-transform the data
        filtered_data = data[column].dropna()
        filtered_data = filtered_data[filtered_data > 0]
        data_filtered = np.log(filtered_data)
    else:
        data_filtered = data[column].dropna()
        data_filtered = data_filtered[data_filtered > 0]

    bin_heights, bin_edges = np.histogram(data_filtered, bins=30, density=True)
    bin_width = bin_edges[1] - bin_edges[0] 

    x = np.linspace(data_filtered.min(), data_filtered.max(), 1000)

    results = []

    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], bin_heights, width=bin_width, alpha=0.5, color="gray", label="Data Histogram")
    
    for dist in distributions:
        # Fit the distribution
        if dist == lognorm or dist == gamma:
            params = dist.fit(data_filtered, floc=0)

        elif dist == beta:
            beta_data = data_filtered * (1 - 2 * epsilon) + epsilon
            params = dist.fit(beta_data)
        else:
            params = dist.fit(data_filtered)

        pdf = dist.pdf(x, *params)
        log_likelihood = np.sum(dist.logpdf(data_filtered, *params))

        # Calculate the AIC
        k = len(params)
        aic = 2 * k - 2 * log_likelihood

        results.append({"Distribution": dist.name, "Log Likelihood": log_likelihood, "AIC": aic})

        plt.plot(x, pdf, label=f"{dist.name} Fit")

    plt.title(f"Fitted Distributions for {title}")
    if with_log_transform:
        plt.xlabel(f"Log-transformed {column}")
    else:
        plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    results_df = pd.DataFrame(results).sort_values(by="AIC")
    print("Fitting Results:")
    print(results_df)

def plot_log_difference_means(df_movies, df_remakes, df_originals, colors, title1="All movies", title2="Remakes", title3="Originals", output_file="log_difference_means.html"):
    """
    Create a bar plot of the mean of log(adjusted_revenue) - log(adjusted_budget) for each dataset.

    Parameters:
        df_movies (pd.DataFrame): The dataset containing all movies.
        df_remakes (pd.DataFrame): The dataset containing remakes.
        df_originals (pd.DataFrame): The dataset containing originals.
        colors (dict): Dictionary of colors for each dataset.
        output_file (str): The filename to save the interactive plot as an HTML file.

    Returns:
        None. Saves the plot as an HTML file and displays it.
    """
    df_movies["log_diff"] = np.log(df_movies["adjusted_revenue"]) - np.log(df_movies["adjusted_budget"])
    df_remakes["log_diff"] = np.log(df_remakes["adjusted_revenue"]) - np.log(df_remakes["adjusted_budget"])
    df_originals["log_diff"] = np.log(df_originals["adjusted_revenue"]) - np.log(df_originals["adjusted_budget"])

    mean_movies = df_movies["log_diff"].mean()
    mean_remakes = df_remakes["log_diff"].mean()
    mean_originals = df_originals["log_diff"].mean()

    datasets = [title1, title2, title3]
    means = [mean_movies, mean_remakes, mean_originals]
    dataset_colors = [colors[title1], colors[title2], colors[title3]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=datasets,
        y=means,
        marker=dict(color=dataset_colors),
        text=[f"{val:.2f}" for val in means],
        textposition="outside",
        name="Mean Log Difference"
    ))

    fig.update_layout(
        title="Mean Log(Revenue) - Log(Budget) Across Datasets",
        xaxis=dict(title="Dataset"),
        yaxis=dict(title="Mean Log Difference"),
        template="plotly_white",
        autosize=True,
        height=600,
        width=800
    )
    fig.write_html(output_file, auto_open=True, include_plotlyjs="cdn")
    print(f"Interactive plot saved to {output_file}")

    fig.show()

def plot_log_revenue_budget_histograms(df_movies, df_remakes, df_originals, colors, output_file="log_revenue_budget_histograms.html"):
    """
    Create interactive histograms of log(adjusted_revenue) - log(adjusted_budget) for each dataset.

    Parameters:
        df_movies (pd.DataFrame): The dataset containing all movies.
        df_remakes (pd.DataFrame): The dataset containing remakes.
        df_originals (pd.DataFrame): The dataset containing originals.
        colors (dict): Dictionary of colors for each dataset.
        output_file (str): The filename to save the interactive plots as an HTML file.

    Returns:
        None. Saves the plots as an HTML file and displays them.
    """
    def compute_log_difference(df):
        return np.log(df["adjusted_revenue"]) - np.log(df["adjusted_budget"])

    df_movies["log_diff"] = compute_log_difference(df_movies)
    df_remakes["log_diff"] = compute_log_difference(df_remakes)
    df_originals["log_diff"] = compute_log_difference(df_originals)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_movies["log_diff"].dropna(),
        name="All Movies",
        marker=dict(color=colors['whole_dataset']),
        opacity=0.7,
        histnorm="percent"
    ))

    fig.add_trace(go.Histogram(
        x=df_remakes["log_diff"].dropna(),
        name="Remakes",
        marker=dict(color=colors['remakes']),
        opacity=0.7,
        histnorm="percent"
    ))

    fig.add_trace(go.Histogram(
        x=df_originals["log_diff"].dropna(),
        name="Originals",
        marker=dict(color=colors['originals']),
        opacity=0.7,
        histnorm="percent"
    ))

    fig.update_layout(
        title="Scale of difference between revenue and budget for datasets",
        xaxis=dict(title="scaled difference (log(revenue) - log(budget))"),
        yaxis=dict(title="Percentage (%)"),
        barmode="overlay",  # Overlay the histograms
        legend=dict(title="Dataset"),
        template="plotly_white",
        autosize=True,
        height=600,  
        width=800    
    )

    fig.write_html(output_file, auto_open=True, include_plotlyjs="cdn")
    print(f"Interactive histograms saved to {output_file}")

    fig.show()
################################################## Statistics across years ####################################################
def plot_histogram(df, column_name, bins, title, ax):
    movie_years_only = df[column_name].dropna().astype(int)
    
    ax.hist(movie_years_only, bins=bins, alpha=0.5)
    ax.set_xlabel("Movie year")
    ax.set_ylabel(f"Number of {title}")
    ax.set_title(title)

def fit_yearly_distribution(df, column_name, title, ax):    
    years = df[column_name].dropna().astype(int)

    lower_bound = np.percentile(years, 2.5)
    upper_bound = np.percentile(years, 97.5)

    filtered_years = years[(years >= lower_bound) & (years <= upper_bound)]

    counts, edges = np.histogram(filtered_years, bins=30, range=(1900, 2020))
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Filter non-zero counts for log-log fit
    nonzero_mask = counts > 0
    nonzero_counts = counts[nonzero_mask]
    nonzero_bin_centers = bin_centers[nonzero_mask]

    log_bin_centers = np.log10(nonzero_bin_centers)
    log_counts = np.log10(nonzero_counts)

    slope, intercept, r_value, p_value, std_err = linregress(log_bin_centers, log_counts)
    fitted_line = slope * log_bin_centers + intercept

    ax.hist(filtered_years, bins=30, range=(1900, 2020), alpha=0.5, log=True, label='Filtered Histogram')
    ax.plot(nonzero_bin_centers, nonzero_counts, 'o', label='Filtered Data (log-log)')
    ax.plot(10**log_bin_centers, 10**fitted_line, 'r-', label=f'Power-Law Fit (slope={slope:.2f})')

    ax.set_yscale('log')
    ax.set_xlabel('Movie Year')
    ax.set_ylabel('Number of Movies (log scale)')
    ax.set_title('Log-Log Plot of Year Counts with Power-Law Fit (95% CI) for ' + title)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    print(f"Results for {title}")
    print(f"Slope: {slope:.2f}")
    print(f"Intercept: {intercept:.2f}")
    print(f"R-squared: {r_value**2:.2f}")

    # theoretical distribution for years 1900-2019
    year_range = np.arange(1900, 2020)
    theoretical_counts = 10**(intercept) * (year_range**slope)
    total = np.sum(theoretical_counts)
    theoretical_pmf = theoretical_counts / total
    theoretical_cdf = np.cumsum(theoretical_pmf)

    def power_law_cdf(x):
        x = np.asarray(x)
        cdf_values = np.zeros_like(x, dtype=float)

        # x < 1900 => CDF = 0
        mask_less = (x < 1900)
        cdf_values[mask_less] = 0.0

        # x >= 2020 => CDF = 1
        mask_greater = (x >= 2020)
        cdf_values[mask_greater] = 1.0

        # For values between 1900 and 2019
        mask_middle = (~mask_less) & (~mask_greater)
  
        indices = np.floor(x[mask_middle]).astype(int) - 1900

        indices = np.clip(indices, 0, len(theoretical_cdf)-1)
        cdf_values[mask_middle] = theoretical_cdf[indices]

        return cdf_values

    # Run KS test
    ks_stat, ks_pvalue = kstest(filtered_years, power_law_cdf)

    alpha = 0.05
    reject_null = ks_pvalue < alpha

    print(f"K-S statistic: {ks_stat:.4f}")
    print(f"K-S p-value: {ks_pvalue:.4f}")
    if reject_null:
        print("Conclusion: Reject H0 at 5% significance level. The data do NOT fit the power-law distribution well.")
    else:
        print("Conclusion: Fail to reject H0 at 5% significance level. The data may be consistent with the power-law distribution.")
    print("="*50)


def fit_yearly_distribution_plotly(df, column_name, title):
    years = df[column_name].dropna().astype(int)

    lower_bound = np.percentile(years, 2.5)
    upper_bound = np.percentile(years, 97.5)
    filtered_years = years[(years >= lower_bound) & (years <= upper_bound)]

    counts, edges = np.histogram(filtered_years, bins=30, range=(1900, 2020))
    bin_centers = (edges[:-1] + edges[1:]) / 2

    nonzero_mask = counts > 0
    nonzero_counts = counts[nonzero_mask]
    nonzero_bin_centers = bin_centers[nonzero_mask]

    log_bin_centers = np.log10(nonzero_bin_centers)
    log_counts = np.log10(nonzero_counts)

    slope, intercept, r_value, p_value, std_err = linregress(log_bin_centers, log_counts)
    fitted_line = slope * log_bin_centers + intercept

    fitted_bin_centers = 10**log_bin_centers
    fitted_counts = 10**fitted_line


    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        name="Histogram",
        opacity=0.5,
        marker=dict(color="blue"),
    ))

    fig.add_trace(go.Scatter(
        x=nonzero_bin_centers,
        y=nonzero_counts,
        mode="markers",
        name="Data (log-log)",
        marker=dict(color="orange", size=8),
    ))

    # power-law fit line
    fig.add_trace(go.Scatter(
        x=fitted_bin_centers,
        y=fitted_counts,
        mode="lines",
        name=f"Power-Law Fit (slope={slope:.2f}) for {title}",
        line=dict(color="red", dash="solid"),
    ))

    fig.update_layout(
        title=f"Log-Log Plot of Year Counts with Power-Law Fit (95% CI) for {title}",
        xaxis=dict(title="Movie Year", type="linear"),
        yaxis=dict(title="Number of Movies (log scale)", type="log"),
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
    )
    return fig


################################################## Runtimes ######################################################
def plot_hist_for_runtime(df, column_name, title, upper_bound, ax):
    filtered_data = df[df[column_name] < (upper_bound)][column_name].dropna()

    ax.hist(filtered_data, bins=50, alpha=0.5)
    ax.set_xlabel("Movie runtime")
    ax.set_ylabel("Number of movies")
    ax.set_title(title + f" (UB={upper_bound})")

def plot_boxplot_with_bound_combined(df, column_name, title, upper_bound, color):
    """
    Generate a boxplot for the filtered data with a specified color.

    Parameters:
        df (pd.DataFrame): Dataframe containing the data.
        column_name (str): Column to create the boxplot for.
        title (str): Title of the boxplot.
        upper_bound (int): Upper bound for filtering data.
        color (str): Color of the boxplot.

    Returns:
        go.Box: A Plotly boxplot trace.
    """
    lower_bound = np.percentile(df[column_name].dropna(), 2.5)
    upper_bound = np.percentile(df[column_name].dropna(), 97.5)
    filtered_data = df[df[column_name] < upper_bound][column_name].dropna()

    boxplot = go.Box(
        x=filtered_data,
        name=f"{title.replace('_',' ')} (UB={upper_bound})",  # Set the name of the boxplot
        boxmean=True, 
        orientation="h", 
        marker=dict(color=color) 
    )

    first_quartile = filtered_data.quantile(0.25)
    third_quartile = filtered_data.quantile(0.75)
    stats = filtered_data.aggregate(["mean", "median", "std", "min", "max"])

    print(f"Statistics for {title} (UB={upper_bound}):")
    print(f"First quartile: {first_quartile}")
    print(f"Third quartile: {third_quartile}")
    print(stats)
    print("=" * 50)

    return boxplot


def test_with_distribution(runtimes, title, ax, dist="norm", x_label='Movie Runtime', y_label='Density', bins=50):
    lower_bound = np.percentile(runtimes, 2.5)
    upper_bound = np.percentile(runtimes, 97.5)
    filtered_runtimes = runtimes[(runtimes >= 40) & (runtimes <= 160)]
    x = np.linspace(filtered_runtimes.min(), filtered_runtimes.max(), 100)

    if dist == "norm":
        # Fit a normal distribution to the runtimes
        mu, std = norm.fit(filtered_runtimes)
        # Perform the Kolmogorov-Smirnov test for goodness-of-fit
        D, p_value = kstest(filtered_runtimes, 'norm', args=(mu, std))
        pdf_fitted = norm.pdf(x, mu, std)
    elif dist == "lognorm":
        # Fit a lognormal distribution to the log of the runtimes
        log_data = np.log(filtered_runtimes)
        mu, std = norm.fit(log_data)
        shape = std
        scale = np.exp(mu)
        loc = 0
        # Perform the Kolmogorov-Smirnov test for goodness-of-fit
        D, p_value = kstest(filtered_runtimes, 'lognorm', args=(shape, loc, scale))
        pdf_fitted = lognorm.pdf(x, s=shape, scale=scale)

    ax.hist(filtered_runtimes, bins=bins, density=True, alpha=0.5, label='Histogram')
    ax.plot(x, pdf_fitted, 'r-', label='Fitted Normal PDF', linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    print(f"Results for {title}:")
    print(f"Kolmogorov-Smirnov D statistic: {D:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value > 0.05:
        print("Conclusion: The data follows a normal distribution (fail to reject H0).")
    else:
        print("Conclusion: The data does not follow a normal distribution (reject H0).")
    print("="*50)

################################################## Genres ######################################################
def pie_chart_data(df, type_of_data, top_n=5):
    """
    Cleans and filters genres, returns names and values for the pie chart.
    """
    df_copy = df.copy()
    df_copy["movie_genres"] = df_copy[type_of_data].apply(cleaning_genres)
    df_genres_exploded = df_copy[type_of_data].dropna().str.split(", ", expand=True).stack().str.strip()

    counts = df_genres_exploded.value_counts()
    frequent_genres = counts[:top_n].index
    df_genres_exploded_filtered = df_genres_exploded[df_genres_exploded.isin(frequent_genres)]

    names = df_genres_exploded_filtered.value_counts().index
    values = df_genres_exploded_filtered.value_counts().values
    return names, values

def plot_genre_trends_by_decade(df, dataset_title, top_n_genres=10, min_year=1900, max_year=2021, cmap="RdBu"):
    """
    Plots a heatmap of the most frequent movie genres over decades for a given dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing movie information.
        dataset_title (str): The title of the dataset (e.g., "Originals" or "Rest").
        top_n_genres (int): The number of top genres to include in the heatmap.
        min_year (int): The minimum year to include in the analysis.
        max_year (int): The maximum year to include in the analysis.
        cmap (str): Colormap to use for the heatmap (default is "RdBu").

    Returns:
        None. Displays the heatmap plot.
    """
    df_movies_copy = df.copy()
    df_movies_copy['movie_genres'] = df_movies_copy['movie_genres'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x
    )

    df_movies_exploded = df_movies_copy.explode("movie_genres")

    counts = df_movies_exploded['movie_genres'].value_counts()
    frequent_genres = counts[counts > 0].index
    df_movies_exploded_filtered = df_movies_exploded[df_movies_exploded['movie_genres'].isin(frequent_genres)]

    df_movies_exploded_filtered.loc[:, 'decade'] = (df_movies_exploded_filtered['year'] // 10) * 10

    df_decades = df_movies_exploded_filtered[
        (df_movies_exploded_filtered['decade'] >= min_year) & (df_movies_exploded_filtered['decade'] <= max_year)
    ]

    genre_counts = df_decades.groupby(['decade', 'movie_genres']).size().unstack(fill_value=0)

    top_genres = genre_counts.sum().nlargest(top_n_genres).index
    genre_counts_top = genre_counts[top_genres]

    plt.figure(figsize=(12, 8))
    sns.heatmap(genre_counts_top, cmap=cmap, annot=True, fmt="d", cbar_kws={'label': 'Count'})
    plt.title(f"Top {top_n_genres} Film Genres from {min_year} to {max_year} (every 10 years) for {dataset_title}")
    plt.xlabel("Genre")
    plt.ylabel("Decade")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_genre_heatmap(df_movies_total, plot_type, top_n_genres=20, output_file="heatmap.html", color_scale="purples"):
    """
    Generate heatmaps for genre relationships between originals and remakes.

    Parameters:
        df_movies_total (pd.DataFrame): The dataset containing movie information.
        plot_type (str): The type of heatmap to generate. Options:
            - "scaled_counts": Min-Max scaled counts of genre pairs.
            - "mean_vote_averages": Mean vote averages for remakes.
            - "log_vote_counts": Log-transformed vote counts.
            - "weighted_vote_average": Weighted vote averages using vote count and vote average.
        top_n_genres (int): The number of top genres to include in the heatmap (default is 20).
        output_file (str): The output HTML file for the interactive heatmap.

    Returns:
        None. Saves the interactive heatmap as an HTML file.
    """
    df_valid = df_movies_total.dropna(subset=['original_wikidata_id', 'remake_wikidata_id'])

    merge_columns = ['wikidata_id', 'movie_genres']
    if plot_type in ["mean_vote_averages", "weighted_vote_average"]:
        merge_columns.append('vote_average')
    if plot_type in ["log_vote_counts", "weighted_vote_average"]:
        merge_columns.append('vote_count')

    df_merged = df_valid.merge(
        df_movies_total[merge_columns],
        left_on='original_wikidata_id',
        right_on='wikidata_id',
        suffixes=('_remake', '_original')
    ).drop(columns='wikidata_id_original')

    df_merged.rename(columns={'movie_genres_remake': 'remake_genres', 'movie_genres_original': 'original_genres'}, inplace=True)

    def explode_genres(df, genre_col):
        df_exploded = df.copy()
        df_exploded[genre_col] = df_exploded[genre_col].str.split(',')
        return df_exploded.explode(genre_col)

    df_exploded = explode_genres(df_merged, 'original_genres')
    df_exploded = explode_genres(df_exploded, 'remake_genres')

    df_exploded['original_genres'] = df_exploded['original_genres'].str.strip()
    df_exploded['remake_genres'] = df_exploded['remake_genres'].str.strip()
    df_exploded = df_exploded.dropna(subset=['original_genres', 'remake_genres'])

    all_genres = pd.concat([df_exploded['original_genres'], df_exploded['remake_genres']])
    top_genres = all_genres.value_counts().head(top_n_genres).index.tolist()

    df_exploded_filtered = df_exploded[
        df_exploded['original_genres'].isin(top_genres) & df_exploded['remake_genres'].isin(top_genres)
    ]
    
    if plot_type == "scaled_counts":
        genre_counts = df_exploded_filtered.groupby(['original_genres', 'remake_genres']).size().reset_index(name='count')
        genre_pivot = genre_counts.pivot(index='original_genres', columns='remake_genres', values='count').fillna(0)
        scaler = MinMaxScaler()
        heatmap_data = pd.DataFrame(
            scaler.fit_transform(genre_pivot),
            index=genre_pivot.index,
            columns=genre_pivot.columns
        )
        title = "Min-Max Scaled Counts of Top Genres (Originals -> Remakes)"

    elif plot_type == "mean_vote_averages":
        genre_vote_averages = df_exploded_filtered.groupby(['original_genres', 'remake_genres'])['vote_average_remake'].mean().reset_index()
        heatmap_data = genre_vote_averages.pivot(index='original_genres', columns='remake_genres', values='vote_average_remake').fillna(0)
        title = "Mean Vote Averages (Remakes) by Genre (Original -> Remakes)"

    elif plot_type == "log_vote_counts":
        df_exploded_filtered = df_exploded_filtered.dropna(subset=['vote_count_remake'])
        genre_vote_counts = df_exploded_filtered.groupby(['original_genres', 'remake_genres'])['vote_count_remake'].mean().reset_index()
        genre_vote_counts['vote_count_remake'] = genre_vote_counts['vote_count_remake'].apply(lambda x: np.log(x) if x > 0 else 0)
        heatmap_data = genre_vote_counts.pivot(index='original_genres', columns='remake_genres', values='vote_count_remake').fillna(0)
        title = "Log-Transformed Vote Counts by Genre (Original -> Remakes)"

    elif plot_type == "weighted_vote_average":
        genre_vote_averages = df_exploded_filtered.groupby(['original_genres', 'remake_genres'])[['vote_average_remake', 'vote_count_remake']].apply(
            lambda x: (x['vote_average_remake'] * x['vote_count_remake']).sum() / x['vote_count_remake'].sum()
        ).reset_index(name='weighted_average')
        heatmap_data = genre_vote_averages.pivot(index='original_genres', columns='remake_genres', values='weighted_average').fillna(0)
        title = "Weighted Vote Averages by Genre (Original -> Remakes)"

    else:
        raise ValueError("Invalid plot_type. Choose from 'scaled_counts', 'mean_vote_averages', 'log_vote_counts', or 'weighted_vote_average'.")

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Remake Genres", y="Original Genres", color=plot_type.replace("_", " ").title()),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale=color_scale, 
        title=title
    )

    fig.update_layout(
    title=title,
    showlegend=False,
    autosize=True,
    template="plotly_white",
    height=600,
    width=800
    )

    pio.write_html(fig, file=output_file, auto_open=True, include_plotlyjs="cdn")
    print(f"Interactive heatmap saved to {output_file}")

    fig.show()

################################################## Ratings ######################################################
def plot_people_perception(df_movies, df_remakes, df_originals, df_rest, column_name, colors, bins=None, is_log=False):
    counts_whole, edges_whole = np.histogram(df_movies[column_name].dropna(), bins=bins)
    counts_remakes, edges_remakes = np.histogram(df_remakes[column_name].dropna(), bins=bins)
    counts_originals, edges_originals = np.histogram(df_originals[column_name].dropna(), bins=bins)
    counts_rest, edges_rest = np.histogram(df_rest[column_name].dropna(), bins=bins)

    bin_centers_whole = (edges_whole[:-1] + edges_whole[1:]) / 2
    bin_centers_remakes = (edges_remakes[:-1] + edges_remakes[1:]) / 2
    bin_centers_originals = (edges_originals[:-1] + edges_originals[1:]) / 2
    bin_centers_rest = (edges_rest[:-1] + edges_rest[1:]) / 2

    nonzero_whole = counts_whole > 0
    nonzero_remakes = counts_remakes > 0
    nonzero_originals = counts_originals > 0
    nonzero_rest = counts_rest > 0

    bin_centers_whole = bin_centers_whole[nonzero_whole]
    bin_centers_remakes = bin_centers_remakes[nonzero_remakes]
    bin_centers_originals = bin_centers_originals[nonzero_originals]
    bin_centers_rest = bin_centers_rest[nonzero_rest]

    counts_whole = counts_whole[nonzero_whole]
    counts_remakes = counts_remakes[nonzero_remakes]
    counts_originals = counts_originals[nonzero_originals]
    counts_rest = counts_rest[nonzero_rest]

    def min_max_scale(arr):
        return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() != arr.min() else arr

    scaled_counts_whole = min_max_scale(counts_whole)
    scaled_counts_remakes = min_max_scale(counts_remakes)
    scaled_counts_originals = min_max_scale(counts_originals)
    scaled_counts_rest = min_max_scale(counts_rest)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers_whole, scaled_counts_whole, marker='o', linestyle='-', label='Scaled whole dataset', color=colors['whole_dataset'])
    plt.plot(bin_centers_remakes, scaled_counts_remakes, marker='o', linestyle='-', label='Scaled remakes', color=colors['remakes'])
    plt.plot(bin_centers_originals, scaled_counts_originals, marker='o', linestyle='-', label='Scaled originals', color=colors['originals'])
    plt.plot(bin_centers_rest, scaled_counts_rest, marker='o', linestyle='-', label='Scaled rest', color=colors['rest'])

    plt.xlabel(f'{column_name.replace("_", " ")} scores (log space)' if is_log else f'{column_name.replace("_", " ")} scores', fontsize=12)
    plt.ylabel('Scaled Counts (log scale)', fontsize=12)
    plt.title(f'Log of Min-Max Scaled Counts of Different {column_name.replace("_", " ")} Scores', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_violin_plot(df_movies, df_remakes, df_originals, output_file="violin_plot.html"):
    """
    Create an interactive violin plot of log-transformed vote counts for all movies, remakes, and originals.

    Parameters:
        df_movies (pd.DataFrame): The dataset containing all movies.
        df_remakes (pd.DataFrame): The dataset containing remakes.
        df_originals (pd.DataFrame): The dataset containing originals.
        output_file (str): The filename to save the interactive plot as an HTML file.

    Returns:
        None. Saves the plot as an HTML file and displays it.
    """

    df_movies['group'] = 'All Movies'
    df_remakes['group'] = 'Remakes'
    df_originals['group'] = 'Originals'

    df_combined = pd.concat([df_movies[['vote_count', 'group']],
                             df_remakes[['vote_count', 'group']],
                             df_originals[['vote_count', 'group']]])

    # Remove rows with zero or NaN vote counts
    df_combined = df_combined[df_combined['vote_count'] > 0]
    df_combined['log_vote_count'] = np.log10(df_combined['vote_count'])

    fig = px.violin(
        df_combined,
        x="group",
        y="log_vote_count",
        color="group",
        box=True,  
        points=None, 
        hover_data={"log_vote_count": True, "vote_count": True},
        title="Interactive Violin Plot of Log Vote Counts by Group",
        labels={"log_vote_count": "Log(Vote Counts)", "group": "Group"}
    )

    # Customize layout
    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Group",
        yaxis_title="Log(Vote Counts)",
        legend_title="Group",
        autosize=True,
        height=600,
        width=800
    )

    fig.write_html(output_file)
    print(f"Violin plot saved as {output_file}")

    fig.show()

################################################## Sentiment ######################################################
def plot_beta_distribution_with_sentiment(df, sentiment_column, a=5, b=0.6, bins=100, output_file=None):
    """
    Plots a Beta distribution and overlays it with a histogram of sentiment scores from the dataset.
    Additionally, calculates the RMSE between the Beta distribution and the histogram data.

    Parameters:
        df (pd.DataFrame): The dataset containing sentiment scores.
        sentiment_column (str): The name of the column containing sentiment scores.
        a (float): Shape parameter `a` for the Beta distribution (default is 5).
        b (float): Shape parameter `b` for the Beta distribution (default is 0.6).
        bins (int or str): Number of bins or binning strategy for the histogram (default is 100).
        output_file (str, optional): File name to save the plot. If None, the plot is displayed.

    Returns:
        float: The calculated RMSE value.
    """
    rv = beta(a, b)
    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100000)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='Beta PDF')

    counts, edges = np.histogram(df[sentiment_column], bins=bins, range=(0, 1))
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Overlay the histogram
    ax.hist(df[sentiment_column], density=True, bins=bins, histtype='stepfilled', alpha=0.2, label='Data Histogram')
    ax.set_xlim([x[0], x[-1]])
    ax.legend(loc='best', frameon=False)

    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Plot saved as {output_file}")
    else:
        plt.show()

    # Calculate RMSE
    beta_pdf = rv.pdf(bin_centers)
    rmse = np.sqrt(np.mean((beta_pdf - counts / counts.sum()) ** 2))
    print(f"RMSE for the sentiment score: {rmse}")

    return rmse

def plot_normalized_sentiment_analysis(df_movies, df_remakes, df_originals, colors, output_file="normalized_sentiment_analysis.html"):
    """
    Create interactive normalized bar plots for sentiment label analysis for all movies, remakes, and originals.

    Parameters:
        df_movies (pd.DataFrame): The dataset containing all movies.
        df_remakes (pd.DataFrame): The dataset containing remakes.
        df_originals (pd.DataFrame): The dataset containing originals.
        colors (dict): Dictionary of colors for each dataset.
        output_file (str): The filename to save the interactive plots as an HTML file.

    Returns:
        None. Saves the plots as an HTML file and displays them.
    """
    sentiment_labels_analysis = df_movies["sentiment_label"].value_counts(normalize=True) * 100
    sentiment_labels_analysis_remakes = df_remakes["sentiment_label"].value_counts(normalize=True) * 100
    sentiment_labels_analysis_originals = df_originals["sentiment_label"].value_counts(normalize=True) * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sentiment_labels_analysis.index,
        y=sentiment_labels_analysis.values,
        name="All Movies",
        marker=dict(color=colors['whole_dataset']),
        text=[f"{val:.2f}%" for val in sentiment_labels_analysis.values],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=sentiment_labels_analysis_remakes.index,
        y=sentiment_labels_analysis_remakes.values,
        name="Remakes",
        marker=dict(color=colors['remakes']),
        text=[f"{val:.2f}%" for val in sentiment_labels_analysis_remakes.values],
        textposition="outside"
    ))

    # Bar plot for the originals dataset
    fig.add_trace(go.Bar(
        x=sentiment_labels_analysis_originals.index,
        y=sentiment_labels_analysis_originals.values,
        name="Originals",
        marker=dict(color=colors['originals']),
        text=[f"{val:.2f}%" for val in sentiment_labels_analysis_originals.values],
        textposition="outside"
    ))

    fig.update_layout(
        title="Normalized Sentiment Label Analysis Across Datasets",
        xaxis=dict(title="Sentiment Labels"),
        yaxis=dict(title="Percentage (%)"),
        barmode="group", 
        legend=dict(title="Dataset"),
        template="plotly_white",
        autosize=True,
        height=600,
        width=800 
    )

    fig.write_html(output_file, auto_open=True, include_plotlyjs="cdn")
    print(f"Interactive normalized sentiment analysis plot saved to {output_file}")

    fig.show()