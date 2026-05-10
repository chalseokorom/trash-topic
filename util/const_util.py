"""Constants for use in review analysis"""

DATA_FILE_PATH = "data/reviews_2026_04_24.csv"

EMBEDDING_PATH = "sentence-transformers/all-mpnet-base-v2"

# llama-cpp variables - generative AI
LLAMA_REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
LLAMA_FILE_PATH = "Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"

LLAMA_PROMPT = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant that creates concise, professional labels for clusters of documents.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    I have a topic described by these keywords:
    [KEYWORDS]

    Sample documents from this topic:
    [DOCUMENTS]

    Based on the information above, provide a single, short topic label (1-5 words). 
    Return ONLY the label. Do not include quotes, intros, or explanations.
    Label: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

# Guided BERTopic varibles
review_topics = [
    # Topic: Service Reliability/Consistency
    ["missed", "time", "times", "extra", "pickup", "pickups"],

    # Topic: Billing/Account Support
    ["help", "helped", "account", "answered", "questions", "issue", "bill",
        "pay", "payment", "phone", "spoke", "called", "representative"],

    # Topic: Company/Employee Reputation/Appreciation
    ["employees", "company", "work", "job", "people", "experience", "team"]
]

waste_stop_words = [
    # Company name (+ mispellings)
    "murrays", "murray", "murrey", "murreys"

    # Waste general words
    "trash", "trash", "garbage", "recycling", "waste", "disposal", "service",

    # Customer service general words
    "help", "helpful", "helped", "thank", "thanks", "customer", "issue",
    "representative", "employee", "employees",
]


# Presidio variables - Name Entity Recognition (NER)
presidio_config = [{"lang_code": "en", "model_name": "en_core_web_lg"}]

entity_mapping = dict(
    PER="PERSON",
    LOC="LOCATION",
    GPE="LOCATION",
    ORG="ORGANIZATION"
)
