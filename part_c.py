
from pathlib import Path
import json
import re
import time
import random

import pandas as pd

import mock_llm

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("Part C setup successful!")
print("Data folder:", DATA_DIR)
print("Output folder:", OUTPUT_DIR)

# Load support tickets
TICKETS_FILE = DATA_DIR / "raw" / "tickets.csv"

tickets = pd.read_csv(TICKETS_FILE, keep_default_na=False)

print("\nTickets loaded successfully!")
print("Total tickets:", len(tickets))
print("Column names:", list(tickets.columns))
print("\nFirst 5 tickets:")
print(tickets.head())

# Check empty and long tickets
MAX_TICKET_LENGTH = 500

tickets["text"] = tickets["text"].astype(str)

tickets["ticket_length"] = tickets["text"].str.len()

empty_tickets = tickets["text"].str.strip().eq("")
long_tickets = tickets["ticket_length"] > MAX_TICKET_LENGTH

print("\nTicket validation results:")
print("Empty tickets:", int(empty_tickets.sum()))
print("Long tickets:", int(long_tickets.sum()))
print("Tickets ready for classification:", int((~empty_tickets & ~long_tickets).sum()))

# Privacy protection: redact PII before sending text to the model
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PHONE_PATTERN = r"(?<!\w)(?:\+?\d[\d\s()-]{7,}\d)(?!\w)"
ORDER_PATTERN = r"\bORD[- ]?\d+\b"


def redact_pii(text):
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(PHONE_PATTERN, "[PHONE]", text)
    text = re.sub(ORDER_PATTERN, "[ORDER_ID]", text, flags=re.IGNORECASE)
    return text


# Quick privacy test
sample_text = "Contact tom@example.com about order ORD-105035"
redacted_text = redact_pii(sample_text)

print("\nPrivacy test:")
print("Original:", sample_text)
print("Redacted:", redacted_text)

# Classification settings
VALID_CATEGORIES = set(mock_llm.CATS)
MAX_RETRIES = 3


def classify_ticket(ticket_text, few_shot=False):
    safe_text = redact_pii(ticket_text)

    system_prompt = (
        "CLASSIFY the customer support ticket. "
        "Return JSON with category and reason. "
        "Allowed categories: billing, delivery, technical, fraud."
    )

    if few_shot:
        system_prompt += (
            "\nEXAMPLES:\n"
            "Unauthorized transaction -> fraud\n"
            "Wrong invoice amount -> billing\n"
            "App error -> technical\n"
            "Late package -> delivery"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": safe_text},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = mock_llm.chat(messages)
            break
        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES - 1:
                return {
                    "status": "flagged_invalid_output",
                    "category": "",
                    "tokens_used": 0,
                }

            delay = 0.05 * (1 + random.uniform(-0.15, 0.15))
            print(f"Retry {attempt}: waiting {delay:.4f} seconds")
            time.sleep(delay)

    # Read response text and token usage
    response_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    try:
        result = json.loads(response_text)
        category = result.get("category", "")

        if category not in VALID_CATEGORIES:
            return {
                "status": "flagged_invalid_output",
                "category": "",
                "tokens_used": tokens_used,
            }

        return {
            "status": "classified_json",
            "category": category,
            "tokens_used": tokens_used,
        }

    except (json.JSONDecodeError, AttributeError, TypeError):
        words = re.findall(r"[a-z]+", response_text.lower())

        for category in VALID_CATEGORIES:
            if category in words:
                return {
                    "status": "classified_keyword",
                    "category": category,
                    "tokens_used": tokens_used,
                }

        return {
            "status": "flagged_invalid_output",
            "category": "",
            "tokens_used": tokens_used,
        }


def run_classification(data, few_shot=False, token_budget=100000):
    results = []
    total_tokens = 0

    for _, row in data.iterrows():
        text = str(row["text"])

        result = {
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "status": "",
            "category": "",
            "tokens_used": 0,
        }

        # Empty and long tickets must not call the model
        if not text.strip():
            result["status"] = "skipped_empty"

        elif len(text) > MAX_TICKET_LENGTH:
            result["status"] = "flagged_long"

        # Reserve enough tokens before making a model call
        elif total_tokens + 100 > token_budget:
         result["status"] = "budget_exhausted"

        else:
            # Privacy check before the model call
            safe_text = redact_pii(text)

            assert not re.search(EMAIL_PATTERN, safe_text)
            assert not re.search(PHONE_PATTERN, safe_text)
            assert not re.search(ORDER_PATTERN, safe_text, re.IGNORECASE)

            # Call classification function
            prediction = classify_ticket(text, few_shot=few_shot)

            result["status"] = prediction["status"]
            result["category"] = prediction["category"]
            result["tokens_used"] = prediction["tokens_used"]

            total_tokens += prediction["tokens_used"]

        results.append(result)

    return pd.DataFrame(results), total_tokens


print("\nRunning zero-shot classification on all tickets...")
zero_shot_results, zero_shot_tokens = run_classification(
    tickets, few_shot=False, token_budget=100000
)

zero_shot_path = OUTPUT_DIR / "triage_zero_shot.csv"
zero_shot_results.to_csv(zero_shot_path, index=False)

print("Zero-shot results saved:", zero_shot_path)
print("Tickets processed:", len(zero_shot_results))
print("Total tokens:", zero_shot_tokens)
print("\nStatus counts:")
print(zero_shot_results["status"].value_counts())

# Evaluate only tickets successfully classified
def calculate_accuracy(results):
    classified = results[
        results["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    if len(classified) == 0:
        return 0.0, 0, 0

    correct = (
        classified["category"] == classified["true_category"]
    ).sum()

    accuracy = correct / len(classified)

    return accuracy, int(correct), len(classified)


accuracy, correct_count, classified_count = calculate_accuracy(
    zero_shot_results
)

print("\nZero-shot evaluation:")
print("Correct predictions:", correct_count)
print("Classified tickets:", classified_count)
print(f"Accuracy: {accuracy:.1%}")
print(
    "Tokens per classified ticket:",
    round(zero_shot_tokens / classified_count, 2)
    if classified_count else 0
)

# Run few-shot classification on all tickets
print("\nRunning few-shot classification on all tickets...")

few_shot_results, few_shot_tokens = run_classification(
    tickets, few_shot=True, token_budget=100000
)

few_shot_path = OUTPUT_DIR / "triage_few_shot.csv"
few_shot_results.to_csv(few_shot_path, index=False)

few_accuracy, few_correct, few_classified = calculate_accuracy(
    few_shot_results
)

print("Few-shot results saved:", few_shot_path)
print("Total tokens:", few_shot_tokens)
print("Correct predictions:", few_correct)
print("Classified tickets:", few_classified)
print(f"Accuracy: {few_accuracy:.1%}")
print(
    "Tokens per classified ticket:",
    round(few_shot_tokens / few_classified, 2)
    if few_classified else 0
)

# Zero-shot confusion table
classified_zero = zero_shot_results[
    zero_shot_results["status"].isin(
        ["classified_json", "classified_keyword"]
    )
]

confusion = pd.crosstab(
    classified_zero["true_category"],
    classified_zero["category"],
    rownames=["Actual"],
    colnames=["Predicted"],
)

print("\nZero-shot confusion table:")
print(confusion)

# Token budget demonstration
print("\nRunning token budget demo...")
budget_results, budget_tokens = run_classification(
    tickets,
    few_shot=False,
    token_budget=5000,
)

budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
budget_results.to_csv(budget_path, index=False)

print("Budget demo saved:", budget_path)
print("Token budget: 5000")
print("Tokens used:", budget_tokens)
print(
    "Budget-exhausted tickets:",
    int((budget_results["status"] == "budget_exhausted").sum())
)

# Create final ticket triage results using few-shot predictions
print("\nCreating final triage results...")

SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"
# Part B output is stored in outputs/
if not SEGMENTS_FILE.exists():
    SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

segments = pd.read_csv(SEGMENTS_FILE)

# Keep only the customer details needed for routing
segment_lookup = segments.set_index("customer_id").to_dict("index")

final_rows = []

for _, row in few_shot_results.iterrows():
    customer_id = row["customer_id"]
    status = row["status"]
    category = row["category"]

    customer = segment_lookup.get(customer_id, {})
    segment_name = customer.get("segment_name", "")
    is_premium = segment_name == "Premium"
    

    if status not in ("classified_json", "classified_keyword"):
        priority = "NORMAL"
        route = "human_review"
        category = ""
    elif category == "fraud":
        priority = "HIGH"
        route = "security_team"
    elif is_premium:
        priority = "HIGH"
        route = "premium_desk"
    else:
        priority = "NORMAL"
        route = "support_queue"

    final_rows.append({
        "ticket_id": row["ticket_id"],
        "customer_id": customer_id,
        "status": status,
        "category": category,
        "segment_name": segment_name,
        "priority": priority,
        "route": route,
        "tokens_used": row["tokens_used"],
    })

final_results = pd.DataFrame(final_rows)

final_path = OUTPUT_DIR / "triage_results.csv"
final_results.to_csv(final_path, index=False)

print("Final results saved:", final_path)
print("\nRouting counts:")
print(final_results["route"].value_counts())
print("\nPriority counts:")
print(final_results["priority"].value_counts())
print("\nFinal output preview:")
print(final_results.head(10))
print("\nTotal final tickets:", len(final_results))
