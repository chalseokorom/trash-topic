"""Constants for use in review analysis"""

DATA_FILE_PATH = "data/murrays_disposal_google_reviews_2026_04_24.csv"

# Feed in guided BERTopic seed topics
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
    "recycling", "waste", "disposal",

    # Employee/customer names
    "rachel", "shari", "jon", "danielle", "daniella", "schleif", "steph", "donna", 
    "mackenzie", "cassie", "lambert", "denise", "bobbiejo",

    # Customer service general words
    "help", "helpful", "helped", "thank", "thanks", "customer", "issue", 
    "representative", "employee", "employees",

    # Time
    "time", "times", "day", "days", "weeks", "months", "years", "always", "every",
    "still", "today", "extra",

    # Communication
    "called", "get", "work", "job", "would", "told", "said", "speaking", "talked", 
    "back", "phone", "issue", "issues", "talking", "question", "questions",

    # Pos Sentiment Noise
    "great", "good", "best", "beyond", "truly", "pleasure", "highly", "recommend", 
    "experience", "amazing", "excellent", "reliable", "friendly", "appreciate", 
    "thanks", "polite", "really", "pleasant", "courteous"
]

# topic_labels = {
#     -1: "Noise + Service Failures",
#     0: "Service Reliability",
#     1: "Account Support",
#     2: "Staff Appreciation",
#     3: "Brand/Customer Loyalty",
# }
