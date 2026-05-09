"""Constants for use in review analysis"""

DATA_FILE_PATH = "data/reviews_2026_04_24.csv"

LLAMA_PATH = "zephyr-7b-alpha.Q4_K_M.gguf"
EMBEDDING_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# Guided BERTopic varibles
review_topics = [
    # Topic: Service Reliability/Consistency
    ["missed", "time", "times", "extra", "pickup", "pickups"],

    # Topic: Billing/Account Support
    ["help", "helped", "account", "answered", "questions", "issue", "bill",
        "pay", "payment", "phone" "spoke", "called", "representative"],

    # Topic: Company/Employee Reputation/Appreciation
    ["employees", "company", "work", "job", "people", "experience", "team"]
]

waste_stop_words = [
    # Company name (+ mispellings)
    "murrays", "murray", "murrey", "murreys", "company", "team",

    # Waste general words
    "trash", "driver", "drivers", "truck", "trucks", "trash", "garbage",
    "recycling", "waste", "disposal", "service",

    # Customer service general words
    "help", "helpful", "helped", "thank", "thanks", "customer", "issue",
    "representative", "employee", "employees",

    # Time
    "time", "times", "day", "days", "weeks", "months", "years", "always", "every",
    "still", "today", "extra",

    # Communication
    "called", "work", "job", "told", "said", "speaking", "talked",
    "phone", "issue", "issues", "talking", "question", "questions",

    # Positive Sentiment Noise
    "great", "good", "best", "beyond", "truly", "pleasure", "highly", "recommend",
    "experience", "amazing", "excellent", "appreciate", "thanks", "polite", "really", "pleasant", "courteous"
]

# topic_labels = {
#     -1: "Noise + Service Failures",
#     0: "Service Reliability",
#     1: "Account Support",
#     2: "Staff Appreciation",
#     3: "Brand/Customer Loyalty",
# }

# Name Entity Recognition (NER) variables:
presidio_config = [{"lang_code": "en", "model_name": "en_core_web_lg"}]

entity_mapping = dict(
    PER="PERSON",
    LOC="LOCATION",
    GPE="LOCATION",
    ORG="ORGANIZATION"
)
