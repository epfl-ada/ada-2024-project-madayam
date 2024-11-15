# Can we Film that Again S'IL VOUS PLAIT? Unveiling Trends in Movie Remakes

| ![generated using GPT/DALL-E](cover_1.webp) |
|:--:|
| *generated using GPT/DALL-E* |

## Abstract

The film industry has long been captivated by the allure of remaking classic movies, offering fresh takes on beloved stories to new generations. The motivation behind this project stems from a curiosity about the cyclical nature of storytelling in cinema and the industry's reliance on nostalgia. This project aims to delve deep into the phenomenon of movie remakes by analyzing films spanning decades to uncover patterns that explain why certain movies are chosen for remakes, their differences from their originals, and the factors contributing to their success or failure. By examining genre popularity, revenue, critical reception, and temporal gaps, we seek to tell the story of how and why the film industry reinvents existing narratives. This analysis will not only highlight the evolving tastes of audiences but also provide insights into the decisions made by filmmakers and studios. Perhaps as history repeats itself, so do pivotal movies of the generations.

## Research Objectives

We briefly outline the research questions that we aim to address in this project. We categorize our questions into two main categories as follows.

### Patterns of Movie to be Remade

What key features or metrics make an original movie more likely to be remade? Explicitly, we are interested in:

1. Whether the genre of the movie, the sentiment of the plot line (which could be determined by inference of LLM model), the time gap between the original and the remake, the popularity, revenue, and critical reception are contributing factors to the likelihood of a movie being remade.

2. How these factors can contribute to a movie being remade more than once?

Overall, we can formalize this section as follows: "What is the difference in the distribution of the movies that are remade and the movies that are not remade?"

### Differences between Original and Remake

In this part, we are interested in comparing the original movie with its remakes. We are explicitly interested in:

3. Comparing the ethnicity and popularity of the cast and the crew, the genre, the critical reception, and the revenue of the movie.

4. The time gap between the original and the remake.

5. The contextual similarities such as historical events may co-occur with the original and remade versions.

To conclude, we can summarise this section as follows: "What is the joint/difference between the distribution of the original movie and the remakes?"

## Datasets

Our project focuses on the CMU Movie Summary Corpus, a dataset that includes information about movies and their characters. However, we require data on movie remakes, and the CMU dataset has to be cleaned and enriched for our task, as described below.

### Wikipedia Movie Remakes

We extracted data from Wikipedia's "List of film remakes" pages (1, 2) and organized it into a tabular format. Afterward, to integrate the crawled data into the base CMU dataset, we retrieve the corresponding WikidataIDs for each movie using its URL.

### TMDB

To address the limitations of the CMU dataset—such as missing values, missing records for remade movies, and its limited feature set—we enhanced it using an offline version of TMDB data from Kaggle. This enrichment process involved:

- Adding missing records related to remade movies.
- Incorporating additional features (e.g., budget).
- Filling gaps in the data using TMDB values.

Additionally, we identified and corrected noisy records in the CMU dataset by cross-referencing them with TMDB, resulting in a more consistent and reliable dataset.

### Google Knowledge Graph

To map the Freebase code to the ethnicities, we get an API key through Google Cloud to access the Google Knowledge Graph to retrieve the ethnicity names and descriptions through their Freebase code.

## Methods

To come up with a sound and reasonable story, we have to take several steps as described in this section.

### Working with Temporal Data

Working with time data collected over a long period requires special attention because the criteria change over time.

1. **Accounting for Inflation**: We will adjust the revenue and budget values for inflation to ensure that the comparison is fair across different time periods. We can employ the Consumer Price Index (CPI) to adjust the values.

2. **Demographic Drifts**: We will account for demographic shifts in the target audience by considering the population and changing the economic status of the audience.

3. **Social Drifts**: The taste of a society may change over time and it has reflections on the scores people give to the movies and reviews of critics. We can take this into account by analyzing the temporal data of the movies.

### Regression Analysis

4. **Regression Analysis**: By using comprehensive regression models, we can examine the relationship between various factors and the likelihood of a movie being remade. This helps to uncover the key features that make certain movies more likely to be revisited.

5. **Casual Analysis**: Methods like __Difference-in-Differences__ allow us to measure the causal effects of remaking a movie by comparing changes over time between remade films and their counterparts that were not remade. This approach is critical for isolating the impact of remakes.

### Sentiment Analysis Using Zero-shot Learning with LLMs

6. **Appealing Contexts**: We will analyze the context and sentiment of the original and remade movies to identify whether they may have influenced the decision to remake the film.

7. **Difference in Plots**: We will investigate the differences in the plot of the original and remade movies to identify any changes that may have been made to appeal to the new audiences. This analysis will help us understand how filmmakers adapt classic stories to resonate with contemporary viewers.

## Proposed Timeline

1. Finishing Data Processing and Deal with Temporal Data: `22.11.2024`
2. Regressing and Sentiment Analysis: `10.12.2024`
3. Creating the Final Data Story Blog: `17.12.2024`

## Organization

| Name    | T1 | T2 | T3 | T4 | T5 | T6 | T7 | Blog |
|:-------:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|
| Amir    | ✓  | ✗  | ✗  | ✓  | ✓  | ✗  | ✗  | ✗    |
| Aryan   | ✗  | ✗  | ✓  | ✗  | ✓  | ✗  | ✓  | ✓    |
| Maria   | ✗  | ✗  | ✗  | ✗  | ✗  | ✓  | ✓  | ✓    |
| Matin   | ✗  | ✓  | ✗  | ✓  | ✗  | ✓  | ✗  | ✗    |
| Yiğit   | ✓  | ✓  | ✓  | ✗  | ✗  | ✗  | ✗  | ✓    |
