# Notebooks

## Setting UP Environment

1. Install the requirements

    ```bash
    pip install -r requirements.txt
    ```

2. Install our `config` package

    ```bash
    cd config;
    pip install -e .
    cd ..;
    ```

3. Download the Data

    ```bash
    gdown --folder https://drive.google.com/drive/folders/1NZD1CNek_Sim8oIJls__RSZOYkbCrre1
    ```

## Structure

```bash
.
└── notebooks/
    └── data/
        ├── cmu_data.ipynb                  <- Load CMU data and add their WikidataID (QID)
        ├── movie_remakes_data.ipynb        <- Crawling the Remake Data
        └── tmdb_data.ipynb                 <- Load TMDB data and add their WikidataID (QID)
```
