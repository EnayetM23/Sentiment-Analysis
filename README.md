# Online Review Sentiment Analysis — Flipkart / Twitter

## Project summary
This project applies natural language processing (NLP) and supervised machine learning to 1,000 synthetic Flipkart/Twitter-style product reviews. The pipeline cleans review text, converts it into TF-IDF features using unigrams and bigrams, trains three classifiers (Logistic Regression, Multinomial Naive Bayes, and Linear SVM), evaluates them on the same stratified 20% test set, and visualizes sentiment patterns over time and by source.

## Dataset
`data/online_reviews_sentiment.csv` contains 1,000 labeled reviews with:
- `review_id`
- `review_text`
- `product`
- `sentiment` (positive, negative, neutral)
- `source` (Flipkart, Twitter)
- `date`
- `rating`

The dataset covers January–December 2025 and 15 products. The sentiment distribution is 400 positive, 340 negative, and 260 neutral reviews.

## Repository structure
```text
sentiment-analysis-project/
├── data/
│   └── online_reviews_sentiment.csv
├── notebooks/
│   ├── sentiment_analysis.ipynb
│   └── sentiment_analysis_submission.ipynb
├── outputs/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── overall_sentiment.png
│   ├── trend_over_time.png
│   ├── sentiment_by_source.png
│   ├── model_results.csv
│   ├── confusion_matrix.csv
│   ├── monthly_sentiment.csv
│   └── sentiment_by_source.csv
├── report/
│   ├── sentiment_analysis_report.docx
│   └── sentiment_analysis_report.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run
1. Install Python 3.10+.
2. Create a virtual environment if desired.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Open `notebooks/sentiment_analysis_submission.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
5. Run the cells from top to bottom.

The notebook expects the CSV file at `../data/online_reviews_sentiment.csv` when run from the `notebooks` directory.

## Models
- **Logistic Regression:** a standard linear classifier that works well with sparse TF-IDF text features.
- **Multinomial Naive Bayes:** a probabilistic baseline commonly used for document classification and compatible with non-negative TF-IDF features.
- **Linear SVM:** a margin-based linear classifier that is effective for high-dimensional sparse text representations.

## Key results
All three models produced 1.00 accuracy, weighted precision, weighted recall, and weighted F1-score on the 200-review held-out test set. The confusion matrix for the selected top-ranked model therefore contains no off-diagonal errors.

Because the dataset is synthetic and relatively formulaic, these perfect results should not be interpreted as evidence that the models would achieve perfect performance on unrestricted real-world social-media text. The report discusses this limitation and suggests testing on more diverse real reviews, including sarcasm, spelling variation, slang, and ambiguous sentiment.

## Deliverables
- Deliverables 1–2: implemented in the submission notebook and supported by the files in `outputs/`.
- Deliverable 3: this README, organized repository structure, requirements file, and meaningful Git history.
- Deliverable 4: the report in both Word and PDF formats.
