import numpy as np
import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# from google.colab import drive
# drive.mount('/content/drive')


# df = pd.read_csv('/content/drive/My Drive/SentimentAnalysisAda/enrich_movie_data.csv')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

def sentiment_analysis(df_plots, tokenizer_name, model_name="distilbert-base-uncased-finetuned-sst-2-english", path="/content/drive/My Drive/SentimentAnalysisAda/", device=device):
    # df_plots['original_index'] = df_plots.index
    df_plots['original_index'] = df_plots["index"]

    df_plots = df_plots.sample(frac=1).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer_summary = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model_summary = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6").to(device)

    sentiment_max_length = tokenizer.model_max_length
    summarizer_max_length = tokenizer_summary.model_max_length

    sentiment_results = []

    for idx, summary in enumerate(tqdm(df_plots["movie_summary"]), start=1):
        if pd.isnull(summary):
            print(f"Index {idx}: Appending None due to null summary.")
            sentiment_results.append(None)
            continue
        else:
            # If summary length in characters is <= sentiment_max_length, we do direct sentiment analysis
            if len(summary) <= sentiment_max_length:
                inputs = tokenizer(
                    summary,
                    truncation=True,
                    max_length=sentiment_max_length,
                    return_tensors='pt'
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=-1)
                    pos_score = predictions[:, 1].item()
                    neg_score = predictions[:, 0].item()
                    if pos_score > neg_score:
                        label = 'POSITIVE'
                        score = pos_score
                    else:
                        label = 'NEGATIVE'
                        score = neg_score
                    sentiment_results.append({'label': label, 'score': score})
            else:
                inputs_summary = tokenizer_summary(
                    summary,
                    truncation=True,
                    max_length=summarizer_max_length,
                    return_tensors='pt'
                ).to(device)

                with torch.no_grad():
                    summarized_ids = model_summary.generate(
                        input_ids=inputs_summary['input_ids'],
                        attention_mask=inputs_summary['attention_mask'],
                        max_length=300,
                        min_length=100,
                        do_sample=False
                    )
                summarized_text = tokenizer_summary.decode(
                    summarized_ids[0],
                    skip_special_tokens=True
                )

                inputs = tokenizer(
                    summarized_text,
                    truncation=True,
                    max_length=sentiment_max_length,
                    return_tensors='pt'
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=-1)
                    pos_score = predictions[:, 1].item()
                    neg_score = predictions[:, 0].item()
                    if pos_score > neg_score:
                        label = 'POSITIVE'
                        score = pos_score
                    else:
                        label = 'NEGATIVE'
                        score = neg_score
                    sentiment_results.append({'label': label, 'score': score})

        if idx % 1000 == 0:
          df_checkpoint = df_plots[:idx].copy()
          df_checkpoint['sentiment'] = sentiment_results

          checkpoint_filename = path+f'sentiment_analysis_checkpoint_{idx}.csv'
          df_checkpoint.to_csv(checkpoint_filename, index=False)
          print(f"Checkpoint saved at iteration {idx} to {checkpoint_filename}.")


    df_plots['sentiment'] = sentiment_results

    df_plots_sorted = df_plots.sort_values('original_index').reset_index(drop=True)

    final_filename = path+'sentiment_analysis_final_sorted.csv'
    df_plots_sorted.to_csv(final_filename, index=False)
    print(f"Final sentiment analysis results saved in original order to '{final_filename}'.")

    return sentiment_results

if __name__ == "__main__":
    df_movies = pd.read_csv('enrich_movie_data.csv')
    df_plots = df_movies.copy()
    df_plots = df_plots["movie_summary"].dropna()
    print("Number of available summaries in the enrich dataset:",len(df_plots))
    
    df_movies['sentiment'] = sentiment_analysis(
        df_plots,
        'distilbert-base-uncased',
        "distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
