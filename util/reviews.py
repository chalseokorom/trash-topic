"""Functions for loading, processing analyzing review data"""
import os
import pandas as pd

DATA_FILE_PATH = "data/reviews_2026_04_24.csv"
STAR_REVIEWS_FILE_PATH = "data/star_reviews_trash_service_2026_04_24.csv"
TEXT_REVIEWS_FILE_PATH = "data/text_reviews_trash_service_2026_04_24.csv"

PRESIDIO_MODEL_CONFIG = [{"lang_code": "en", "model_name": "en_core_web_lg"}]

SPACY_ENTITY_MAPPING = dict(
    PER="PERSON",
    LOC="LOCATION",
    GPE="LOCATION",
    ORG="ORGANIZATION"
)


def get_reviews() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create csv for both review types unless the files already exist"""
    if os.path.exists(STAR_REVIEWS_FILE_PATH) and os.path.exists(STAR_REVIEWS_FILE_PATH):
        star_reviews = pd.read_csv(STAR_REVIEWS_FILE_PATH)
        text_reviews = pd.read_csv(TEXT_REVIEWS_FILE_PATH)
        return star_reviews, text_reviews

    text_reviews, star_reviews = load_reviews()

    text_reviews.to_csv(TEXT_REVIEWS_FILE_PATH, index=False)
    star_reviews.to_csv(STAR_REVIEWS_FILE_PATH, index=False)

    return text_reviews, star_reviews


def load_reviews() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read reviews from file, clean data types and split them based on text."""
    reviews = pd.read_csv(DATA_FILE_PATH)

    # Turn date strings into datetime format
    reviews["publishedAtDate"] = reviews["publishedAtDate"].apply(
        pd.to_datetime)
    reviews["responseFromOwnerDate"] = reviews["responseFromOwnerDate"].apply(
        pd.to_datetime)

    # Split data into those with and without text reviews
    # and remove columns no longer relevant for star-only reviews
    mask = reviews.text.isna()
    text_based_columns = [
        "text", "responseFromOwnerText", "responseFromOwnerDate"]
    
    text_reviews = add_text_features(reviews[~mask])
    star_reviews = reviews[mask].drop(columns=text_based_columns)

    return text_reviews, star_reviews


def get_lowest_tfidf_terms(docs: pd.Series, n_terms: int = 20) -> pd.Series:
    """Get the words with the LOWEST average TF-IDF scores"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    weights = tfidf.fit_transform(docs)

    # Words that are common everywhere but specific nowhere
    importance = weights.sum(axis=0).tolist()[0]
    vocabulary = tfidf.get_feature_names_out()
    word_freq = pd.DataFrame({'word': vocabulary, 'score': importance}).sort_values(
        by='score', ascending=False).word

    return word_freq[0:n_terms]


def get_spacy_documents(docs: pd.Series) -> list:
    """Retrieve text as spacy spans"""
    import spacy

    nlp = spacy.load("en_core_web_lg")
    return list(nlp.pipe(docs))


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
                                       model_to_presidio_entity_mapping=SPACY_ENTITY_MAPPING)

    analyzer = AnalyzerEngine(nlp_engine=SpacyNlpEngine(
        models=PRESIDIO_MODEL_CONFIG, ner_model_configuration=ner_config))
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
