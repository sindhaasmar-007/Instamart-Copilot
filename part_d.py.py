import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
CHECKPOINT_FILE = STATE_DIR / "part_d_checkpoint.json"
ORDERS_CSV = CLEAN_DIR / "orders.csv"
SCENARIOS_CSV = RAW_DIR / "agent_scenarios.csv"
POLICY_EVAL_CSV = RAW_DIR / "policy_eval.csv"

POLICY_NAMES = [
    "delivery_policy.txt",
    "escalation_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt",
    "refund_policy.txt",
]
MAX_AGENT_STEPS = 7
MAX_LLM_RETRIES = 3


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Ignore an incomplete final line after an interrupted write.
                    continue
    return rows


def write_jsonl(path, records):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
    temp_path.replace(path)


def append_jsonl(path, record):
    # One complete JSON object per line; append only after all validation passes.
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_policies():
    documents = []
    for filename in POLICY_NAMES:
        path = POLICY_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing policy document: {path}")
        documents.append({"name": filename, "text": path.read_text(encoding="utf-8").strip()})
    return documents


def make_chunks(text, size=40, overlap=10):
    words = text.split()
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks


class PolicyRetriever:
    def __init__(self, documents, overlap=10):
        self.rows = []
        for doc in documents:
            for index, chunk in enumerate(make_chunks(doc["text"], 40, overlap)):
                self.rows.append({"policy": doc["name"], "chunk_id": index, "text": chunk})
        if not self.rows:
            raise ValueError("No policy chunks were loaded.")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([row["text"] for row in self.rows])

    def search(self, query, top_k=3):
        if not query.strip():
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in indices:
            if scores[index] > 0:
                item = dict(self.rows[int(index)])
                item["score"] = round(float(scores[index]), 4)
                results.append(item)
        return results


class SupportAgent:
    def __init__(self):
        self.documents = load_policies()
        self.retriever = PolicyRetriever(self.documents, overlap=10)
        self.no_overlap_retriever = PolicyRetriever(self.documents, overlap=0)
        self.orders = {row["order_id"]: row for row in read_csv(ORDERS_CSV)}
        self.customers = {row["customer_id"]: row for row in read_csv(CLEAN_DIR / "customers.csv")}
        self.tool_trace = []
        self.ledger = read_jsonl(LEDGER_FILE)

    def get_order(self, order_id):
        order = self.orders.get(str(order_id).upper())
        if not order:
            return {"error": "Order not found", "order_id": order_id}
        return {"order_id": order["order_id"], "customer_id": order["customer_id"],
                "status": order["status"], "order_amount": float(order["order_amount"]),
                "delivery_minutes": order["delivery_minutes"]}

    def search_policy(self, query):
        hits = self.retriever.search(str(query), top_k=3)
        if not hits:
            return {"found": False, "message": "No relevant policy text found."}
        return {"found": True, "chunks": [
            {"policy": hit["policy"], "chunk_id": hit["chunk_id"], "text": hit["text"],
             "score": hit["score"]} for hit in hits
        ]}

    def calculate_refund(self, order_id=None, order_amount=None, delivery_minutes=None):
        order = self.orders.get(str(order_id).upper()) if order_id else None
        if order:
            status = order["status"].strip().lower()
            amount = float(order["order_amount"])
            raw_minutes = order["delivery_minutes"]
        else:
            status = "completed"
            try:
                amount = float(order_amount)
                raw_minutes = delivery_minutes
            except (TypeError, ValueError):
                return {"outcome": "FALLBACK", "refund_inr": 0.0,
                        "reason": "Required order details are missing."}
        if status != "completed":
            return {"outcome": "DENIED", "refund_inr": 0.0,
                    "reason": "Only completed orders can be refunded."}
        try:
            minutes = float(raw_minutes)
        except (TypeError, ValueError):
            return {"outcome": "FALLBACK", "refund_inr": 0.0,
                    "reason": "Delivery details are incomplete; human review is needed."}
        if minutes <= 45:
            return {"outcome": "DENIED", "refund_inr": 0.0,
                    "reason": "Delivery took 45 minutes or less."}
        if amount > 1000:
            return {"outcome": "ESCALATE", "refund_inr": round(amount, 2),
                    "reason": "Refund above INR 1000 requires human approval."}
        return {"outcome": "REFUND", "refund_inr": round(amount, 2),
                "reason": "Completed order was delivered after 45 minutes."}

    def issue_refund(self, order_id, refund_inr):
        order = self.orders.get(str(order_id).upper())
        if not order:
            return {"issued": False, "reason": "Order not found."}
        if order["status"].strip().lower() != "completed":
            return {"issued": False, "reason": "Order is not completed."}
        try:
            amount = round(float(refund_inr), 2)
            order_amount = float(order["order_amount"])
        except (TypeError, ValueError):
            return {"issued": False, "reason": "Invalid refund amount."}
        if amount <= 0 or amount > order_amount or amount > 1000:
            return {"issued": False, "reason": "Invalid amount or human approval required."}

        # Idempotency: never issue a second refund for the same order.
        current_ledger = read_jsonl(LEDGER_FILE)
        existing = [row for row in current_ledger
                    if row.get("order_id") == order["order_id"]
                    and row.get("status") == "ISSUED"]
        if existing:
            return {"issued": False, "duplicate": True,
                    "reason": "A refund is already recorded for this order.",
                    "existing_refund_inr": existing[-1].get("refund_inr", 0)}

        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario_id": getattr(self, "current_scenario_id", ""),
            "order_id": order["order_id"],
            "customer_id": order["customer_id"],
            "refund_inr": amount,
            "status": "ISSUED",
            "payment_method": "original_payment_method"
        }
        append_jsonl(LEDGER_FILE, entry)
        return {"issued": True, **entry}

    def escalate_to_human(self, order_id, refund_inr=0, reason="Human review required"):
        return {"outcome": "ESCALATE", "order_id": order_id,
                "refund_inr": float(refund_inr or 0), "reason": reason,
                "queue": "human_support"}

    def _tool_functions(self):
        return {
            "get_order": lambda **a: self.get_order(**a),
            "search_policy": lambda **a: self.search_policy(**a),
            "calculate_refund": lambda **a: self.calculate_refund(**a),
            "issue_refund": lambda **a: self.issue_refund(**a),
            "escalate_to_human": lambda **a: self.escalate_to_human(**a),
        }

    def _chat_with_retry(self, messages):
        for attempt in range(MAX_LLM_RETRIES):
            try:
                return mock_llm.chat(messages, tools=None)
            except mock_llm.TransientLLMError:
                if attempt == MAX_LLM_RETRIES - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def process_scenario(self, scenario):
        scenario_id = scenario["scenario_id"]
        self.current_scenario_id = scenario_id
        user_message = scenario.get("message", "")
        tools = self._tool_functions()
        messages = [
            {"role": "system", "content": (
                "AGENT You are a safe Instamart refund support agent. "
                "Use tools to verify order status and policy before deciding. "
                "Never issue a refund above INR 1000; escalate those requests. "
                "Do not repeat a tool call indefinitely."
            )},
            {"role": "user", "content": user_message}
        ]
        tool_names = []
        retrieved_chunks = 0
        final_text = ""
        outcome = "FALLBACK"
        refund_amount = 0.0
        reason = "Agent reached the safe step limit or could not complete the request."
        seen_calls = set()

        for step in range(1, MAX_AGENT_STEPS + 1):
            try:
                response = self._chat_with_retry(messages)
            except mock_llm.TransientLLMError:
                outcome, reason = "FALLBACK", "Temporary model error persisted after retries; human handoff required."
                break

            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            if choice.finish_reason != "tool_calls" or not getattr(message, "tool_calls", []):
                final_text = content
                upper = content.upper()
                if upper.startswith("REFUND"):
                    outcome = "REFUND"
                    found = re.search(r"INR\s*([0-9]+(?:\.[0-9]+)?)", content, re.I)
                    refund_amount = float(found.group(1)) if found else 0.0
                    reason = content
                elif upper.startswith("ESCALATE"):
                    outcome = "ESCALATE"
                    reason = content
                    order_match = re.search(r"\bORD-\d+\b", user_message, re.I)
                    self.escalate_to_human(order_match.group(0) if order_match else "", reason=content)
                elif upper.startswith("DENIED"):
                    outcome, reason = "DENIED", content
                elif upper.startswith("NOT_FOUND"):
                    outcome, reason = "NOT_FOUND", content
                elif upper.startswith("NEEDS_INFO"):
                    outcome, reason = "FALLBACK", content
                else:
                    outcome, reason = "FALLBACK", content or reason
                break

            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [self._serialize_tool_call(tc) for tc in message.tool_calls]})
            for tool_call in message.tool_calls:
                name = tool_call.name
                raw_args = tool_call.arguments
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                call_key = (name, json.dumps(args, sort_keys=True))
                if call_key in seen_calls:
                    outcome, reason = "FALLBACK", "Repeated tool call detected; safe human handoff."
                    break
                seen_calls.add(call_key)
                if name not in tools:
                    tool_result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        tool_result = tools[name](**args)
                    except (TypeError, ValueError, KeyError) as exc:
                        tool_result = {"error": f"Invalid tool arguments: {exc}"}
                tool_names.append(name)
                if name == "search_policy":
                    chunks = tool_result.get("chunks", []) if isinstance(tool_result, dict) else tool_result
                    retrieved_chunks += len(chunks or [])
                messages.append({"role": "tool", "name": name,
                                 "tool_call_id": getattr(tool_call, "id", ""),
                                 "content": json.dumps(tool_result, ensure_ascii=False)})
            else:
                continue
            break
        else:
            outcome, reason = "FALLBACK", "Maximum agent steps reached; human handoff required."

        # The model's tool call issues the refund. If a previous interrupted run already
        # wrote it to the ledger, issue_refund refuses a duplicate and we recover the result.
        order_match = re.search(r"\bORD-\d+\b", user_message, re.I)
        order_id = order_match.group(0).upper() if order_match else ""
        if outcome == "REFUND":
            ledger = read_jsonl(LEDGER_FILE)
            prior = next((x for x in ledger if x.get("order_id") == order_id
                          and x.get("status") == "ISSUED"), None)
            if prior:
                refund_amount = float(prior.get("refund_inr", refund_amount))
            elif refund_amount <= 0:
                outcome, reason = "FALLBACK", "Model response did not contain a valid refund amount."
            else:
                # Normally issue_refund already appended the entry during the tool call.
                outcome, reason = "FALLBACK", "Refund response lacked a matching ledger entry; human review required."

        if outcome == "ESCALATE":
            calc = self.calculate_refund(order_id=order_id)
            refund_amount = float(calc.get("refund_inr", 0.0))

        return {
            "scenario_id": scenario_id,
            "customer_id": scenario.get("customer_id", ""),
            "order_id": order_id,
            "outcome": outcome,
            "refund_inr": round(refund_amount, 2) if outcome in ("REFUND", "ESCALATE") else 0.0,
            "reason": reason,
            "retrieved_policy_chunks": retrieved_chunks,
            "steps": min(step, MAX_AGENT_STEPS),
            "tool_calls": tool_names,
            "final_response": final_text
        }

    @staticmethod
    def _serialize_tool_call(tool_call):
        args = tool_call.arguments
        if not isinstance(args, str):
            args = json.dumps(args or {})
        return {"id": getattr(tool_call, "id", ""), "type": "function",
                "function": {"name": tool_call.name, "arguments": args}}

    def evaluate_retrieval(self):
        if not POLICY_EVAL_CSV.exists():
            return {"error": "data/raw/policy_eval.csv was not found."}
        examples = read_csv(POLICY_EVAL_CSV)
        report = {"chunk_words": 40, "overlap_words": 10, "no_overlap_words": 0,
                  "queries_evaluated": 0, "overlap_hits": 0, "no_overlap_hits": 0}
        for row in examples:
            query = row.get("question") or row.get("query") or row.get("customer_message") or ""
            expected = (row.get("expected_policy") or row.get("policy") or row.get("policy_file") or "").lower()
            if not query.strip():
                continue
            report["queries_evaluated"] += 1
            overlap_hits = self.retriever.search(query, top_k=3)
            no_overlap_hits = self.no_overlap_retriever.search(query, top_k=3)
            # Recall@3 counts whether the expected policy is present in top 3.
            # If the source dataset has no expected-policy field, report retrieval presence separately.
            if expected:
                if any(expected in item["policy"].lower() for item in overlap_hits):
                    report["overlap_hits"] += 1
                if any(expected in item["policy"].lower() for item in no_overlap_hits):
                    report["no_overlap_hits"] += 1
            else:
                if overlap_hits:
                    report["overlap_hits"] += 1
                if no_overlap_hits:
                    report["no_overlap_hits"] += 1
        denominator = max(1, report["queries_evaluated"])
        report["overlap_recall_at_3"] = round(report["overlap_hits"] / denominator, 4)
        report["no_overlap_recall_at_3"] = round(report["no_overlap_hits"] / denominator, 4)
        return report

    def run(self):
        checkpoint = {}
        if CHECKPOINT_FILE.exists():
            try:
                checkpoint = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                checkpoint = {}
        completed = checkpoint.get("completed", {})
        scenarios = read_csv(SCENARIOS_CSV)
        runs = []

        # Resume safety: checkpoint results are reused, and the ledger independently
        # prevents duplicate refunds if a crash happened after ledger append.
        for scenario in scenarios:
            sid = scenario["scenario_id"]
            if sid in completed:
                result = completed[sid]
            else:
                result = self.process_scenario(scenario)
                result["expected_outcome"] = scenario.get("expected_outcome", "")
                try:
                    result["expected_refund_inr"] = float(scenario.get("expected_refund_inr") or 0)
                except ValueError:
                    result["expected_refund_inr"] = 0.0
                result["match_expected_outcome"] = result["outcome"] == result["expected_outcome"]
                result["match_expected_refund"] = abs(
                    float(result["refund_inr"]) - result["expected_refund_inr"]) < 0.01
                completed[sid] = result
                CHECKPOINT_FILE.write_text(
                    json.dumps({"completed": completed}, indent=2), encoding="utf-8"
                )
            runs.append(result)

        write_jsonl(RUNS_FILE, runs)
        report = self.evaluate_retrieval()
        (OUTPUT_DIR / "retrieval_evaluation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        matches = sum(bool(row.get("match_expected_outcome")) for row in runs)
        summary = {
            "scenarios": len(runs),
            "outcome_matches": matches,
            "outcome_accuracy": round(matches / max(1, len(runs)), 4),
            "refund_ledger_entries": len(read_jsonl(LEDGER_FILE)),
            "refund_ledger": str(LEDGER_FILE.relative_to(ROOT)),
            "agent_runs": str(RUNS_FILE.relative_to(ROOT)),
            "retrieval_evaluation": report,
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # Reset the deterministic mock's transient-error counters for this independent run.
    mock_llm.reset()
    SupportAgent().run()
