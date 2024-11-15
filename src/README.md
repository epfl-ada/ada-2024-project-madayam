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

3. Download the Data: All our processed data is stored in a Google Drive folder. You can download the data using the following command:

    ```bash
    gdown --folder https://drive.google.com/drive/folders/1NZD1CNek_Sim8oIJls__RSZOYkbCrre1
    ```

## Structure

```bash
.
├── notebooks/
│   ├── cmu_data.ipynb                <- Load, parse, and clean CMU data and add their WikidataID (QID).
│   │                                    Load and parse plot summery and char metadata.
│   ├── movie_remakes_data.ipynb      <- Crawling the Remake Data
│   └── tmdb_data.ipynb               <- Load TMDB data and add their WikidataID (QID)
├── result_eda.ipynb                  <- Exploratory Data Analysis
└── result_methods.ipynb              <- Our proof of concept methods
```
