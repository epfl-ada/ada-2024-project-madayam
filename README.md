# Can we Film that Again S'IL VOUS PLAIT? Unveiling Trends in Movie Remakes

| ![generated using GPT/DALL-E](cover_1.webp) |
|:--:|
| *generated using GPT/DALL-E* |

## Abstract

The film industry has long been captivated by the allure of remaking classic movies, offering fresh takes on beloved stories to new generations. The motivation behind this project stems from a curiosity about the cyclical nature of storytelling in cinema and the industry's reliance on nostalgia. This project aims to delve deep into the phenomenon of movie remakes by analyzing films spanning decades to uncover patterns that explain why certain movies are chosen for remakes, their differences from their originals, and the factors contributing to their success or failure. By examining genre popularity, revenue, critical reception, and temporal gaps, we seek to tell the story of how and why the film industry reinvents existing narratives. This analysis will not only highlight the evolving tastes of audiences but also provide insights into the decisions made by filmmakers and studios. Perhaps as history repeats itself, so do pivotal movies of the generations.

## Research Objectives

We briefly outline the research questions that we aim to address in this project. We categorize our questions in three main categories as follows.

### Patterns of Movie to be Remade

What key features or metrics make an original movie more likely to be remade? Explicitly, we are interested in:

1. Whether the genre of the movie, the time gap between the original and the remake, the popularity, revenue and the critical reception are contributing factors to the likelihood of a movie being remade.

2. How these factors can contribute to a movie to be remade more than once.

We can formalize this section as follows: "What is the difference of the distribution of the movie that are remade and the movies that are not remade?"

### Differences between Original and Remake

In this part, we are interested in comparing the original movie with its remakes. We are explicitly interested in:

3. Comparing the ethnicity and popularity of the cast and the crew, the genre, the critical reception, and the revenue of the movie.

4. The time gap between the original and the remake.

5. The contextual drifts that may be in the remade version, which can be determined using LLMs.

We can formalize this section as follows: "What is the joint/difference-of the distribution of the original movie and the remakes?"

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

** MARIA S'IL VOUS PLAIT**

and we utilize the Google Knowledge Graph to retrieve Freebase codes for ethnicities. 


** MATIN AND AMIR S'IL VOUS PLAIT**

## Methods
- Show that the method works well on a simple example

## Proposed timeline
## Organization within the team: A list of internal milestones up until project Milestone P3.
## Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
