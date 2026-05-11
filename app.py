import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
from transformers import pipeline
import plotly.express as px


st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border-left: 6px solid #2563eb;
        text-align: center;
        min-height: 120px;
    }
    .metric-number {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
    }
    .metric-label {
        font-size: 14px;
        color: #6b7280;
    }
    .positive-card {
        border-left: 6px solid #16a34a;
    }
    .neutral-card {
        border-left: 6px solid #f59e0b;
    }
    .negative-card {
        border-left: 6px solid #dc2626;
    }
    .info-box {
        background: #eff6ff;
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #2563eb;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Load model once
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )


sentiment_model = load_sentiment_model()


# -----------------------------
# Helper functions
# -----------------------------
def convert_rating_to_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def get_overall_sentiment(avg_rating):
    if avg_rating >= 3.5:
        return "Positive"
    elif avg_rating <= 2.5:
        return "Negative"
    else:
        return "Neutral"


def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    elif file_name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, uploaded_file.name)

            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)

                    if file.lower().endswith(".csv"):
                        return pd.read_csv(file_path)

                    elif file.lower().endswith(".xlsx") or file.lower().endswith(".xls"):
                        return pd.read_excel(file_path)

        raise ValueError("No CSV or Excel file found inside the ZIP file.")

    else:
        raise ValueError("Unsupported file type. Please upload CSV, Excel, or ZIP.")


def run_sentiment_analysis(df, review_column, max_reviews=300, batch_size=16):
    reviews = df[review_column].dropna().astype(str).str.strip()
    reviews = reviews[reviews != ""]

    if len(reviews) == 0:
        raise ValueError("No usable reviews found in the selected review column.")

    reviews = reviews.head(max_reviews)

    result_df = df.loc[reviews.index].copy()

    predictions = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_reviews = len(reviews)

    for start in range(0, total_reviews, batch_size):
        end = min(start + batch_size, total_reviews)

        batch_reviews = reviews.iloc[start:end].tolist()

        batch_predictions = sentiment_model(
            batch_reviews,
            truncation=True,
            max_length=512
        )

        predictions.extend(batch_predictions)

        progress = end / total_reviews
        progress_bar.progress(progress)
        status_text.text(f"Processed {end} of {total_reviews} reviews...")

    result_df["Predicted Rating"] = [
        int(prediction["label"][0]) for prediction in predictions
    ]

    result_df["Confidence"] = [
        round(prediction["score"], 4) for prediction in predictions
    ]

    result_df["Sentiment"] = result_df["Predicted Rating"].apply(
        convert_rating_to_sentiment
    )

    status_text.text("Sentiment analysis completed.")

    return result_df


def create_summary(df):
    total_reviews = len(df)
    positive_reviews = (df["Sentiment"] == "Positive").sum()
    neutral_reviews = (df["Sentiment"] == "Neutral").sum()
    negative_reviews = (df["Sentiment"] == "Negative").sum()
    average_rating = df["Predicted Rating"].mean()

    return {
        "total": total_reviews,
        "positive": positive_reviews,
        "neutral": neutral_reviews,
        "negative": negative_reviews,
        "average_rating": average_rating,
        "overall_sentiment": get_overall_sentiment(average_rating)
    }


def display_summary_cards(summary):
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{summary["total"]}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card positive-card">
                <div class="metric-number">{summary["positive"]}</div>
                <div class="metric-label">Positive Reviews</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card neutral-card">
                <div class="metric-number">{summary["neutral"]}</div>
                <div class="metric-label">Neutral Reviews</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card negative-card">
                <div class="metric-number">{summary["negative"]}</div>
                <div class="metric-label">Negative Reviews</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{summary["average_rating"]:.1f}</div>
                <div class="metric-label">Average Rating</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{summary["overall_sentiment"]}</div>
                <div class="metric-label">Overall Sentiment</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def create_product_summary(df, product_column):
    product_summary = df.groupby(product_column).agg(
        Total_Reviews=("Sentiment", "count"),
        Positive_Reviews=("Sentiment", lambda x: (x == "Positive").sum()),
        Neutral_Reviews=("Sentiment", lambda x: (x == "Neutral").sum()),
        Negative_Reviews=("Sentiment", lambda x: (x == "Negative").sum()),
        Average_Predicted_Rating=("Predicted Rating", "mean")
    ).reset_index()

    product_summary["Average_Predicted_Rating"] = product_summary[
        "Average_Predicted_Rating"
    ].round(1)

    product_summary["Overall_Sentiment"] = product_summary[
        "Average_Predicted_Rating"
    ].apply(get_overall_sentiment)

    return product_summary


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">Product Review Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">A BERT-based dashboard for analyzing product review sentiment.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    This app can analyze a single pasted review or a full dataset uploaded as CSV, Excel, or ZIP.
    If the dataset contains multiple products, you can search and filter by product name or product ID.
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Review Analysis", "File Upload Dashboard"]
)


# -----------------------------
# Mode 1: Single Review
# -----------------------------
if mode == "Single Review Analysis":
    st.header("Single Review Analysis")

    user_review = st.text_area(
        "Paste a product review:",
        height=150,
        placeholder="Example: The headphones sound great, but the battery life is disappointing."
    )

    if st.button("Analyze Review"):
        if user_review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            prediction = sentiment_model(user_review, truncation=True)[0]
            rating = int(prediction["label"][0])
            sentiment = convert_rating_to_sentiment(rating)

            col1, col2, col3 = st.columns(3)

            col1.metric("Predicted Rating", f"{rating} stars")
            col2.metric("Sentiment", sentiment)
            col3.metric("Confidence", f"{prediction['score']:.2%}")

            st.info(
                f"The model predicts this review as **{sentiment}** with a rating of **{rating} stars**."
            )


# -----------------------------
# Mode 2: File Upload Dashboard
# -----------------------------
else:
    st.header("File Upload Dashboard")

    uploaded_file = st.file_uploader(
        "Upload a CSV, Excel, or ZIP file",
        type=["csv", "xlsx", "xls", "zip"]
    )

    if uploaded_file is not None:
        try:
            raw_df = read_uploaded_file(uploaded_file)

            st.subheader("Dataset Preview")
            st.dataframe(raw_df.head())

            st.subheader("Select Columns")

            columns = raw_df.columns.tolist()

            review_column = st.selectbox(
                "Select the column that contains review text",
                columns
            )

            has_product_column = st.checkbox(
                "My dataset has a product name or product ID column",
                value=True
            )

            product_column = None

            if has_product_column:
                product_column = st.selectbox(
                    "Select the product column",
                    columns
                )

            st.subheader("Analysis Settings")

            available_reviews = raw_df[review_column].dropna().astype(str).str.strip()
            available_reviews = available_reviews[available_reviews != ""]
            available_count = len(available_reviews)

            if available_count == 0:
                st.warning("No usable reviews found in the selected review column.")
            else:
                max_slider_value = min(1000, available_count)
                min_slider_value = min(10, max_slider_value)
                default_value = min(300, max_slider_value)

                max_reviews = st.slider(
                    "Select number of reviews to analyze",
                    min_value=min_slider_value,
                    max_value=max_slider_value,
                    value=default_value,
                    step=10
                )

                st.info(
                    f"This run will analyze {max_reviews} reviews out of {available_count} usable reviews. "
                    "Limiting reviews helps the app run faster on CPU."
                )

                if st.button("Run Sentiment Analysis"):
                    with st.spinner("Running BERT sentiment analysis. Please wait..."):
                        analyzed_df = run_sentiment_analysis(
                            raw_df,
                            review_column,
                            max_reviews=max_reviews,
                            batch_size=16
                        )

                    st.session_state["analyzed_df"] = analyzed_df
                    st.session_state["review_column"] = review_column
                    st.session_state["product_column"] = product_column

                    st.success("Sentiment analysis completed.")

        except Exception as e:
            st.error(f"Error: {e}")

    if "analyzed_df" in st.session_state:
        analyzed_df = st.session_state["analyzed_df"]
        review_column = st.session_state["review_column"]
        product_column = st.session_state.get("product_column")

        st.divider()
        st.header("Overall Dataset Summary")

        overall_summary = create_summary(analyzed_df)
        display_summary_cards(overall_summary)

        st.subheader("Overall Sentiment Distribution")

        sentiment_counts = analyzed_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        color_map = {
            "Positive": "green",
            "Neutral": "orange",
            "Negative": "red"
        }

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map=color_map,
            text="Count",
            title="Positive, Neutral, and Negative Review Counts"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predicted Rating Distribution")

        rating_counts = analyzed_df["Predicted Rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Predicted Rating", "Count"]

        fig2 = px.bar(
            rating_counts,
            x="Predicted Rating",
            y="Count",
            text="Count",
            title="Predicted 1 to 5 Star Rating Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Download Full Results")

        csv_data = analyzed_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download analyzed results as CSV",
            data=csv_data,
            file_name="bert_sentiment_results.csv",
            mime="text/csv"
        )

        if product_column is not None:
            st.divider()
            st.header("Product-Level Sentiment Analysis")

            product_summary = create_product_summary(analyzed_df, product_column)

            st.subheader("Product Summary Table")
            st.dataframe(product_summary)

            product_summary_csv = product_summary.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download product summary as CSV",
                data=product_summary_csv,
                file_name="product_sentiment_summary.csv",
                mime="text/csv"
            )

            st.subheader("Search Product")

            search_keyword = st.text_input(
                "Enter part of a product name or product ID",
                placeholder="Example: mouse, charger, phone"
            )

            if search_keyword.strip() != "":
                filtered_summary = product_summary[
                    product_summary[product_column].astype(str).str.contains(
                        search_keyword,
                        case=False,
                        na=False
                    )
                ]

                if filtered_summary.empty:
                    st.warning("No matching products found.")
                else:
                    st.write("Matching Products")
                    st.dataframe(filtered_summary)

                    selected_product = st.selectbox(
                        "Select one product for detailed analysis",
                        filtered_summary[product_column].astype(str).tolist()
                    )

                    selected_product_df = analyzed_df[
                        analyzed_df[product_column].astype(str) == selected_product
                    ]

                    selected_summary = create_summary(selected_product_df)

                    st.subheader(f"Summary for: {selected_product}")
                    display_summary_cards(selected_summary)

                    selected_sentiment_counts = selected_product_df[
                        "Sentiment"
                    ].value_counts().reset_index()

                    selected_sentiment_counts.columns = ["Sentiment", "Count"]

                    fig3 = px.pie(
                        selected_sentiment_counts,
                        names="Sentiment",
                        values="Count",
                        title=f"Sentiment Distribution for {selected_product}"
                    )

                    st.plotly_chart(fig3, use_container_width=True)

                    st.subheader("Reviews for Selected Product")

                    st.dataframe(
                        selected_product_df[
                            [
                                product_column,
                                review_column,
                                "Predicted Rating",
                                "Confidence",
                                "Sentiment"
                            ]
                        ]
                    )