# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a data-driven customer intelligence and support copilot for an Instamart-style online grocery platform.

It uses customer, product, order, and support ticket data to analyse customer behaviour, identify customer segments, predict customer lapse risk, classify support tickets, and route tickets to appropriate support teams.

## Reproduce

### Requirements

- Python 3.11+
- pandas
- numpy
- scikit-learn
- joblib
- SQLite 3.25+

### Setup

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Generate the project data:

   ```bash
   python generate_data.py
   ```

5. Clean the raw data:

   ```bash
   python clean_data.py
   ```

6. Verify the SQLite database:

   ```bash
   python check_db.py
   ```

7. Run the project scripts from the repository root directory to reproduce the relevant results:

   ```bash
   python part_a.py
   python part_b.py
   python part_c.py
   ```

   Run the scripts in the order required by their dependencies. Check each script's output and error messages.

**Important:** Keep `generate_data.py` and `mock_llm.py` unchanged if the assignment requires these files to remain verbatim.

## Part A — Data Audit and Cleaning

### Data Cleaning

- Audited raw customer, product, and order data.
- Checked missing values, duplicate records, invalid values, and data types.
- Standardised customer city and gender values.
- Cleaned order statuses and timestamps.
- Removed duplicate orders and retained the latest record for each order ID.
- Recovered missing order amounts using product prices, quantities, and discounts where possible.
- Handled invalid and missing delivery times.
- Saved cleaned CSV files in `data/clean/`.
- Created the SQLite database at `data/instamart.db`.
- Added SQL analysis queries in `sql/queries.sql`.

### Part A Validation Results

| Check | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Missing order amounts | 0 |
| Missing completed-order delivery times | 0 |
| Completed order amount total | ₹2,998,668.80 |

## Part B — Customer Intelligence

### Customer Segmentation

- Calculated Recency, Frequency, and Monetary (RFM) features using completed orders.
- Compared K-Means clustering for K values from 2 to 8 using inertia and silhouette scores.
- Selected K = 4 based on the highest silhouette score among the tested values.
- Profiled customer segments using average recency, frequency, and monetary value.
- Identified four customer segments:
  - `Regular_Low_Spend`
  - `Premium`
  - `Loyal_Frequent`
  - `At_Risk_Inactive`
- Used Principal Component Analysis (PCA) to visualise customer segments in two dimensions.
- Saved segmentation results to `outputs/customer_segments.csv`.

### Lapse Prediction

- Defined the prediction cutoff date as 2025-09-01.
- Used completed orders before the cutoff date to calculate customer features.
- Labelled customers as lapsed when they had no completed orders on or after the cutoff date.
- Split eligible customers into training and testing sets using an 80/20 stratified split.
- Built Logistic Regression and Random Forest classification pipelines with preprocessing.
- Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Compared baseline models with models trained using oversampling on the training data only.
- Selected the balanced Logistic Regression model based on test recall and ROC-AUC.
- Saved the trained model to `outputs/lapse_model.joblib`.

### Premium Retention List

- Generated an illustrative retention list for Premium customers using predicted lapse probabilities.
- Saved the results to `outputs/retention_list.csv`.

**Important limitations:** Lapse labels use future order activity relative to the prediction cutoff date. The Premium segment label is based on whole-window RFM data, so the retention list is illustrative and should not be treated as a leakage-free production evaluation.

## Part C — Customer Support Ticket Classification

### Overview

This component classifies customer support tickets into four categories: billing, delivery, technical, and fraud. It includes privacy protection, retry handling, output validation, token tracking, and ticket routing.

### Data Validation and Privacy

- Total tickets processed: 164
- Empty tickets skipped: 4
- Long tickets flagged: 4
- Tickets ready for classification: 156
- Email addresses, phone numbers, and order IDs are redacted before ticket text is sent to the model.
- Empty and long tickets are not sent to the model.

### Classification Results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Correct predictions | 126 | 148 |
| Successfully classified tickets | 147 | 148 |
| Accuracy | 85.7% | 100.0% |
| Total tokens used | 7,996 | 11,584 |
| Tokens per classified ticket | 54.39 | 78.27 |

In this run, few-shot classification achieved higher accuracy, with increased token usage.

### Zero-shot Error Analysis

The zero-shot confusion table showed that some fraud and technical tickets were incorrectly classified as billing. Few-shot examples improved the classification results in this run.

### Reliability and Token Budget

- Transient model errors are retried, up to three attempts per ticket.
- Invalid model outputs are flagged for human review.
- A 5,000-token budget demonstration was performed.
- The budget demonstration used 5,037 tokens because the implementation checks the accumulated token total before each model call. A permitted call can take the total above the configured budget.
- Empty, long, and invalid-output tickets are routed for human review.

### Ticket Routing

The final output contains 164 tickets.

| Route | Ticket count |
|---|---:|
| `support_queue` | 101 |
| `security_team` | 37 |
| `premium_desk` | 10 |
| `human_review` | 16 |
| **Total** | **164** |

- High priority: 47 tickets
- Normal priority: 117 tickets

Fraud tickets are routed to `security_team` with HIGH priority. Successfully classified non-fraud tickets from Premium customers are routed to `premium_desk` with HIGH priority. Other successfully classified tickets go to `support_queue` with NORMAL priority. Empty, long, and invalid-output tickets go to `human_review`.

### Generated Outputs

- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`

### Part C Limitations

- Results come from the provided mock LLM and ticket dataset; they do not establish real-world model performance.
- Few-shot accuracy may vary with different data and model behaviour.
- Human review remains necessary for invalid or unclassifiable tickets.
- The token-budget demonstration can exceed its configured limit by the tokens consumed in the final permitted call.

## Part D — Integration and Evaluation

### Integration

- Customer segments are loaded from `outputs/customer_segments.csv`.
- Few-shot classification results are used to create the final triage file.
- Each ticket is assigned a category, customer segment, priority, and route where applicable.
- Fraud tickets are routed to `security_team` with HIGH priority.
- Premium customers with successfully classified non-fraud tickets are routed to `premium_desk` with HIGH priority.
- Other successfully classified tickets are routed to `support_queue` with NORMAL priority.
- Empty, long, or invalid-output tickets are routed to `human_review`.

### Final Validation

| Validation | Result |
|---|---:|
| Total tickets | 164 |
| Tickets routed to support queue | 101 |
| Tickets routed to security team | 37 |
| Tickets routed to premium desk | 10 |
| Tickets routed to human review | 16 |
| HIGH priority | 47 |
| NORMAL priority | 117 |

The final integrated ticket-routing results are saved in `outputs/triage_results.csv`.

### Evaluation Summary

- **Part A:** Produces cleaned data and a SQLite database for analysis.
- **Part B:** Produces customer segments, a lapse prediction model, and a retention list.
- **Part C:** Compares zero-shot and few-shot ticket classification and demonstrates ticket routing.
- **Part D:** Combines customer segment information with ticket classification results to produce integrated routing decisions.

The reported metrics describe the current dataset and mock-LLM run. They should not be interpreted as proof of production performance.

## Limitations and Next Steps

- Validate data quality and business rules with additional automated tests.
- Evaluate the support copilot against independent ticket examples and the assignment's evaluation scenarios.
- Improve token-budget enforcement so that model calls cannot exceed the configured budget.
- Test classification performance on unseen tickets and monitor errors over time.
- Add clearer error handling, logging, and reproducibility checks.
- Document assumptions and potential data leakage in the lapse prediction and retention workflow.
- Evaluate a production-grade language model before any real-world deployment.
- Keep sensitive customer information protected and maintain human review for uncertain or high-risk tickets.
