# Product Review Sentiment Analysis

This project is a BERT-based product review sentiment analysis web application built using Streamlit.

## Features

- Analyze a single pasted product review
- Upload CSV, Excel, or ZIP files
- Select the review column manually
- Select the product column manually
- View total reviews analyzed
- View positive, neutral, and negative review counts
- View average predicted rating
- View overall sentiment
- View product-level sentiment summaries
- Search products by partial name or product ID
- View charts and visualizations
- Download analyzed results as CSV

## Model Used

The app uses the pretrained Hugging Face model:

`nlptown/bert-base-multilingual-uncased-sentiment`

This model predicts review sentiment as 1 to 5 stars.

## Sentiment Mapping

- 1–2 stars = Negative
- 3 stars = Neutral
- 4–5 stars = Positive

## Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt