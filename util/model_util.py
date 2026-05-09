"""Functions to customize models for BERTopic"""
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from bertopic.representation import LlamaCPP
from llama_cpp import Llama

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

from hdbscan import HDBSCAN
from umap import UMAP

# Custom module
import util.const_util as const_util


def get_hdbscan() -> HDBSCAN:
    """Utilize HDBSCAN to group similar documents into clusters that will become topics"""
    return HDBSCAN(min_cluster_size=15,
                   min_samples=2,
                   prediction_data=True,
                   cluster_selection_epsilon=0.05,
                   cluster_selection_method='leaf',
                   gen_min_span_tree=True)


def get_umap() -> UMAP:
    """Create a UMAP model with a fixed seed for reproducible results when testing"""
    return UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)


def get_tfidf() -> TfidfVectorizer:
    """Utilize TfidfVectorizer to find words that are common everywhere but specific nowhere"""
    return TfidfVectorizer(stop_words='english')


def get_vectorizer() -> CountVectorizer:
    """Utilize CountVectorizer to create human-readble labels"""
    stop_words = list(set(stopwords.words('english')))
    # stop_words.extend(const_util.waste_stop_words)
    return CountVectorizer(stop_words=stop_words, ngram_range=(1, 3), min_df=3)


def get_ctfidf() -> ClassTfidfTransformer:
    """Utilize ClassTfidfTransformer to reduce the impact of words that appear in too many topics"""
    return ClassTfidfTransformer(reduce_frequent_words=True)

def get_llama_rep() -> LlamaCPP:
    """Utilize generative AI (LLama) to generate human-readable labels"""
    llm = Llama(model_path=const_util.LLAMA_PATH, n_ctx=2048)
    return LlamaCPP(llm)

def get_bertopic() -> BERTopic:
    """Utilize BERTopic to create interpretable and semantically meaningful topics"""
    return BERTopic(
        representation_model=get_llama_rep(),
        umap_model=get_umap(),
        hdbscan_model=get_hdbscan(),
        vectorizer_model=get_vectorizer(),
        ctfidf_model=get_ctfidf(),
        seed_topic_list=const_util.review_topics,
        nr_topics='auto'
    )

def export_model(topic_model: BERTopic) -> None:
    """Export the topic model to the /model folder"""
    embedding_model = const_util.EMBEDDING_PATH
    topic_model.save("model/", serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=embedding_model)
