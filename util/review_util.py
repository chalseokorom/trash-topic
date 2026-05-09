"""Functions for processing data and analyzing review text"""
import pandas as pd
import spacy

# Custom modules
import util.const_util as const_util
import util.model_util as model_util


def get_reviews() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read reviews from file, clean data types and split them based on text."""
    reviews = pd.read_csv(const_util.DATA_FILE_PATH)

    # Turn date strings into datetime format
    reviews["publishedAtDate"] = reviews["publishedAtDate"].apply(
        pd.to_datetime)
    reviews["responseFromOwnerDate"] = reviews["responseFromOwnerDate"].apply(
        pd.to_datetime)

    # Split data into those with and without text reviews
    mask = reviews.text.isna()

    # Remove columns no longer relevant for stars-only reviews
    text_based_columns = [
        "text", "responseFromOwnerText", "responseFromOwnerDate"]

    return reviews[~mask], reviews[mask].drop(columns=text_based_columns)


def get_lowest_tfidf_terms(docs: pd.Series, n_terms: int = 20) -> pd.Series:
    """Get the words with the LOWEST average TF-IDF scores"""
    tfidf = model_util.get_tfidf()
    weights = tfidf.fit_transform(docs)

    # These are words that are common everywhere but specific nowhere
    importance = weights.sum(axis=0).tolist()[0]
    vocabulary = tfidf.get_feature_names_out()
    word_freq = pd.DataFrame({'word': vocabulary, 'score': importance}).sort_values(
        by='score', ascending=False).word

    return word_freq[0:n_terms]


def get_spacy_documents(docs: pd.Series) -> list:
    model = spacy.load("en_core_web_sm")
    return list(model.pipe(docs))


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and create spacy documents."""
    df['OwnerResponseDays'] = (df["publishedAtDate"] -
                               df["responseFromOwnerDate"]).dt.days

    df['TextDensity'] = df['text'].apply(
        lambda text: len(text) / len(text.split()))

    df["Document"] = get_spacy_documents(df.text)

    return df
