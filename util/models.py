"""Functions for customizing BERTopic model"""
from bertopic import BERTopic
# import representation

EMBEDDING_PATH = "sentence-transformers/all-mpnet-base-v2"

TRASH_STOP_WORDS = []
REVIEW_TOPICS = []


def get_bertopic() -> BERTopic:
    """Utilize BERTopic to create interpretable and semantically meaningful topics"""
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    from hdbscan import HDBSCAN
    from umap import UMAP

    # Utilize UMAP model with a fixed seed for reproducible results when testing
    umap = UMAP(n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                low_memory=True,
                random_state=0)

    # Utilize HDBSCAN to group similar documents into clusters that will become topics
    hdbscan = HDBSCAN(min_cluster_size=15,
                      min_samples=2,
                      cluster_selection_epsilon=0.05,
                      cluster_selection_method='leaf',
                      prediction_data=True,
                      gen_min_span_tree=True)

    # Utilize CountVectorizer to create human-readble labels
    stop_words = list(set(stopwords.words('english')))
    # stop_words.extend(TRASH_STOP_WORDS)
    vectorizer = CountVectorizer(stop_words=stop_words,
                                 ngram_range=(1, 3),
                                 min_df=3)

    # Utilize ClassTfidfTransformer to reduce the impact of words that appear in too many topics
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    return BERTopic(
        # representation_model=representation.get_representation_model()
        umap_model=umap,
        hdbscan_model=hdbscan,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        # seed_topic_list=REVIEW_TOPICS,
        calculate_probabilities=False,
        top_n_words=10,
        nr_topics=20,
        # verbose=True,
    )


def export_topic_model(topic_model: BERTopic) -> None:
    """Export the topic model to the model folder"""
    topic_model.save("model/", serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=EMBEDDING_PATH)
