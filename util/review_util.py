"""Functions for processing data and analyzing review text"""
from pathlib import Path

import pandas as pd

# Custom modules
import util.const_util as const_util
import util.model_util as model_util


def get_reviews() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create csv for both review types unless the files already exist"""
    star_reviews_file_path = Path(
        'data/star_reviews_trash_service_2026_04_24.csv')
    text_reviews_file_path = Path(
        'data/text_reviews_trash_service_2026_04_24.csv')

    if star_reviews_file_path.exists() and text_reviews_file_path.exists():
        star_reviews = pd.read_csv(star_reviews_file_path)
        text_reviews = pd.read_csv(text_reviews_file_path)
        return star_reviews, text_reviews

    text_reviews, star_reviews = load_reviews()
    text_reviews = add_text_features(text_reviews)

    text_reviews.to_csv(text_reviews_file_path, index=False)
    star_reviews.to_csv(star_reviews_file_path, index=False)

    return star_reviews, text_reviews


def load_reviews() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read reviews from file, clean data types and split them based on text."""
    reviews = pd.read_csv(const_util.DATA_FILE_PATH)

    # Turn date strings into datetime format
    reviews["publishedAtDate"] = reviews["publishedAtDate"].apply(
        pd.to_datetime)
    reviews["responseFromOwnerDate"] = reviews["responseFromOwnerDate"].apply(
        pd.to_datetime)

    # Split data into those with and without text reviews
    mask = reviews.text.isna()

    # Remove columns no longer relevant for star-only reviews
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
    """Retrieve text as spacy spans"""
    import spacy

    model = spacy.load("en_core_web_lg")
    return list(model.pipe(docs))


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and create spacy documents."""
    df['OwnerResponseDays'] = (df["publishedAtDate"] -
                               df["responseFromOwnerDate"]).dt.days

    df['TextDensity'] = df['text'].apply(
        lambda text: len(text) / len(text.split()))

    df['text'] = anonymize_text(df['text'])

    df["Document"] = get_spacy_documents(df.text)

    return df


def anonymize_text(docs: pd.Series) -> list[str]:
    from presidio_analyzer.nlp_engine import SpacyNlpEngine, NerModelConfiguration
    from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine

    from presidio_anonymizer.entities import RecognizerResult as AnonymizerRecognizerResult
    from presidio_anonymizer import BatchAnonymizerEngine

    from typing import List, cast

    """Removes PII from reviews: names, places, and phone numbers"""
    ner_config = NerModelConfiguration(default_score=0.6,
                                       model_to_presidio_entity_mapping=const_util.entity_mapping)

    analyzer = AnalyzerEngine(nlp_engine=SpacyNlpEngine(
        models=const_util.presidio_config, ner_model_configuration=ner_config))
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)
    analyzer_results = batch_analyzer.analyze_iterator(
        docs.tolist(), language='en')

    # Converts a batch of Analyzer results to the format required by the Anonymizer.
    iterator = batch_analyzer.analyze_iterator(docs.tolist(), language='en')
    analyzer_results = [
        [r for r in result] for result in iterator]

    analyzer_results = cast(
        List[List[AnonymizerRecognizerResult]], analyzer_results)

    batch_anonymizer = BatchAnonymizerEngine()
    return batch_anonymizer.anonymize_list(docs.tolist(), analyzer_results)


def view_review_topics(topics: pd.DataFrame) -> None:
    """Print the topics in each class"""
    print("Topics (by Class): ")
    for i in topics.index:
        t = topics.iloc[i]
        print(t.Topic, ": ", t.Words)
