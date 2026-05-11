````md
# Product Review Sentiment Analysis

Live App: https://sentiment-reviewer.streamlit.app/

This project is a web-based product review sentiment analysis application built using **Streamlit** and a pretrained **BERT-based sentiment analysis model** from Hugging Face.

The app can analyze a single product review or process a dataset uploaded as a CSV, Excel, or ZIP file. It predicts whether product reviews are **Positive**, **Neutral**, or **Negative**, and it can also create product-level sentiment summaries.

---

## Features

- Analyze a single pasted product review
- Upload CSV, Excel, or ZIP files
- Select the review column manually
- Select the product name or product ID column manually
- View overall sentiment summary
- View product-level sentiment summary
- Search products using a keyword
- View charts and visualizations
- Download analyzed results as CSV

---

## Model Used

This project uses the pretrained Hugging Face model:

`nlptown/bert-base-multilingual-uncased-sentiment`

The model predicts product review sentiment as a rating from **1 to 5 stars**.

The ratings are converted into sentiment labels:

| Predicted Rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

---

## Project Structure

```text
product-review-sentiment-analysis/
│
├── app.py
├── requirements.txt
└── README.md
````

| File               | Description                |
| ------------------ | -------------------------- |
| `app.py`           | Main Streamlit application |
| `requirements.txt` | Required Python libraries  |
| `README.md`        | Project documentation      |

---

## Requirements

The required libraries are listed in `requirements.txt`.

To install them, run:

```bash
pip install -r requirements.txt
```

The main libraries used are:

* `streamlit`
* `transformers`
* `torch`
* `torchvision`
* `pandas`
* `openpyxl`
* `plotly`

---

## Run Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/product-review-sentiment-analysis.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Go Into the Project Folder

```bash
cd product-review-sentiment-analysis
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the App

```bash
streamlit run app.py
```

After running the command, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Run in GitHub Codespaces

You can also run the project directly in GitHub Codespaces.

### Step 1: Open Codespaces

From the GitHub repository:

```text
Code → Codespaces → Create codespace on main
```

### Step 2: Install Dependencies

In the Codespaces terminal, run:

```bash
pip install -r requirements.txt
```

### Step 3: Run the App

```bash
streamlit run app.py
```

### Step 4: Open the App

In Codespaces, open the forwarded port:

```text
Ports → 8501 → Open in Browser
```

Do not use the localhost link directly in Codespaces.

---

## Deploy on Streamlit Community Cloud

This app can be deployed using Streamlit Community Cloud.

### Step 1: Go to Streamlit Community Cloud

```text
https://share.streamlit.io
```

Sign in with GitHub.

### Step 2: Create a New App

Click:

```text
New app
```

Then choose the GitHub repository.

### Step 3: Use These Settings

```text
Repository: YOUR_USERNAME/product-review-sentiment-analysis
Branch: main
Main file path: app.py
```

### Step 4: Deploy

Click **Deploy**.

Streamlit will install the dependencies from `requirements.txt` and run the app.

---

## Dataset Format

The uploaded dataset should contain at least:

* a column containing review text
* a column containing product name or product ID

Example dataset:

| product_name      | review_content                                     |
| ----------------- | -------------------------------------------------- |
| Wireless Mouse    | The mouse works very well and feels comfortable.   |
| Phone Charger     | The charger stopped working after two days.        |
| Bluetooth Speaker | Sound quality is good but battery life is average. |

After uploading a file, the app lets the user choose the correct review column and product column.

---

## Supported File Types

The app supports:

* `.csv`
* `.xlsx`
* `.xls`
* `.zip`

If a ZIP file is uploaded, the app looks for a CSV or Excel file inside it.

---

## Notes

* The first time the app runs, the BERT model may take some time to load.
* Large datasets may take longer to process.
* The app includes a slider to limit the number of reviews analyzed.
* For best performance, test with around 100–300 reviews first.
* The confidence score shows how sure the model is, but it does not guarantee that the prediction is correct.

---

