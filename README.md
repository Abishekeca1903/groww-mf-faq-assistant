# Groww Mutual Fund Facts Assistant — Milestone 4

A facts-only Retrieval-Augmented Generation (RAG) FAQ prototype for public mutual-fund scheme information. **Groww is the selected product context.** The knowledge corpus is scoped to HDFC Mutual Fund. This project does not connect to Groww accounts, execute transactions, or access investor data.

## Scope

- **Product context:** Groww
- **AMC:** HDFC Mutual Fund
- **Schemes:** HDFC Large Cap Fund; HDFC Flexi Cap Fund; HDFC ELSS Tax Saver Fund; HDFC Nifty 50 Index Fund
- **Approved publishers:** HDFC AMC, SEBI and AMFI official public pages/documents
- **Source inventory:** 22 candidate official URLs (also intended for `data/sources.csv`)

> **Source validation note:** This is a candidate source inventory, not proof that every URL is live or was successfully crawled. Verify the latest official document before treating a changing scheme value as current.

## Features

- Factual FAQ topics: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark and statement instructions.
- RAG retrieval using embeddings and FAISS.
- OpenAI API (`gpt-4o-mini`) answer generation.
- One official source link for each factual answer.
- Refusal for investment advice, portfolio questions and performance comparisons.
- Basic PII-pattern rejection for PAN, Aadhaar, email, phone, OTP and account/folio identifiers.
- Maximum two answer sentences plus citation.

## Official source list (22 candidate URLs)

| ID | Source | Publisher | Link |
|---|---|---|---|
| S01 | HDFC Large Cap Fund — Regular Plan | HDFC AMC | [Open](https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/regular) |
| S02 | HDFC Large Cap Fund — Direct Plan | HDFC AMC | [Open](https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct) |
| S03 | HDFC Flexi Cap Fund — Direct Plan | HDFC AMC | [Open](https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct) |
| S04 | HDFC ELSS Tax Saver Fund — Regular Plan | HDFC AMC | [Open](https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/regular) |
| S05 | HDFC Nifty 50 Index Fund — Regular Plan | HDFC AMC | [Open](https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-50-index-fund/regular) |
| S06 | HDFC Nifty 50 Index Fund — Direct Plan | HDFC AMC | [Open](https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-50-index-fund/direct) |
| S07 | HDFC Mutual Fund Factsheets | HDFC AMC | [Open](https://www.hdfcfund.com/mutual-funds/factsheets) |
| S08 | HDFC SIP FAQs | HDFC AMC | [Open](https://www.hdfcfund.com/services/faqs/systematic-investment-plan) |
| S09 | HDFC MF Factsheet — September 2026 | HDFC AMC | [PDF](https://files.hdfcfund.com/s3fs-public/2026-10/HDFC%20MF%20Factsheet%20-%20September%202026.pdf) |
| S10 | HDFC MF Index Factsheet — September 2026 | HDFC AMC | [PDF](https://files.hdfcfund.com/s3fs-public/2026-10/HDFC%20MF%20Index%20Factsheet%20-%20September%202026.pdf) |
| S11 | HDFC ELSS Tax Saver Fund Facts — October 2026 | HDFC AMC | [PDF](https://files.hdfcfund.com/s3fs-public/Others/2026-10/Fund%20Facts%20-%20HDFC%20ELSS%20-%20Tax%20Saver%20Fund.pdf) |
| S12 | HDFC Statement of Additional Information and Addendum | HDFC AMC | [PDF](https://files.hdfcfund.com/s3fs-public/2026-10/SAI%20with%20updated%20Addendum%20dated%20September%2030%2C%202026_0.pdf) |
| S13 | SEBI — Understanding the Riskometer | SEBI | [Open](https://investor.sebi.gov.in/riskometer.html) |
| S14 | SEBI Circular — Riskometer and portfolio disclosure | SEBI | [PDF](https://www.sebi.gov.in/web/?file=https%3A%2F%2Fwww.sebi.gov.in%2Fsebi_data%2Fattachdocs%2Fapr-2021%2F1619691350204.pdf) |
| S15 | SEBI Investor — Consolidated Account Statement | SEBI | [Open](https://investor.sebi.gov.in/consolidated_account_statement.html) |
| S16 | AMFI — Download CAS | AMFI | [Open](https://www.amfiindia.com/online-center/download-cas) |
| S17 | AMFI — Consolidated Account Statement | AMFI | [Open](https://www.amfiindia.com/investor/become-mf-distributor?zoneName=consolidatedAcct) |
| S18 | AMFI — Investor Service FAQs | AMFI | [Open](https://www.amfiindia.com/investor/become-mf-distributor?zoneName=InvestorService) |
| S19 | HDFC Mutual Fund — Request Statement | HDFC AMC | [Open](https://www.hdfcfund.com/services/additional-info/request-statement) |
| S20 | HDFC Mutual Fund — Consolidated Account Statement | HDFC AMC | [Open](https://www.hdfcfund.com/services/consolidated-account-statement) |
| S21 | HDFC Mutual Fund — Capital Gains Statement Guide | HDFC AMC | [Open](https://www.hdfcfund.com/learn/blog/how-get-capital-gain-statement-mutual-fund-schemes-india) |
| S22 | HDFC Flexi Cap Fund Factsheet — August 2026 | HDFC AMC | [PDF](https://files.hdfcfund.com/s3fs-public/2026-09/HDFC%20MF%20Factsheet%20-%20August%202026.pdf) |

A machine-readable list is intended at `data/sources.csv`.

## Sample Q&A (8 examples)

These are sample test cases and expected behavior. The assistant must retrieve current changing values from the latest official source; it must not guess.

### 1. What is the lock-in period for HDFC ELSS Tax Saver Fund?
**Expected answer:** ELSS investments have a three-year lock-in period, calculated separately from the allotment date of each instalment.

**Source:** [HDFC ELSS Tax Saver Fund scheme page](https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/regular)

### 2. What is the exit load for HDFC Nifty 50 Index Fund?
**Expected answer:** Retrieve and state the latest exit-load clause from the scheme page or SID. If it cannot be verified, say so rather than guessing.

**Source:** [HDFC Nifty 50 Index Fund scheme page](https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-50-index-fund/regular)

### 3. What is the benchmark of HDFC Nifty 50 Index Fund?
**Expected answer:** State the benchmark exactly as specified in the latest official scheme document or factsheet, including TRI designation if applicable.

**Source:** [HDFC Nifty 50 Index Fund scheme page](https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-50-index-fund/regular)

### 4. What is the minimum SIP for HDFC ELSS Tax Saver Fund?
**Expected answer:** Retrieve the current SIP minimum and instalment conditions from the latest official scheme document/addendum; do not guess.

**Source:** [HDFC SAI and addendum](https://files.hdfcfund.com/s3fs-public/2026-10/SAI%20with%20updated%20Addendum%20dated%20September%2030%2C%202026_0.pdf)

### 5. What is the minimum SIP for HDFC Flexi Cap Fund?
**Expected answer:** Retrieve the current SIP minimum and frequency/instalment conditions from the latest official scheme document/addendum; do not guess.

**Source:** [HDFC SAI and addendum](https://files.hdfcfund.com/s3fs-public/2026-10/SAI%20with%20updated%20Addendum%20dated%20September%2030%2C%202026_0.pdf)

### 6. How can I request a capital-gains statement?
**Expected answer:** Summarize the current request steps from HDFC AMC's official statement service page. Do not request personal account details in chat.

**Source:** [HDFC AMC — Request Statement](https://www.hdfcfund.com/services/additional-info/request-statement)

### 7. What does the mutual-fund Riskometer mean?
**Expected answer:** The Riskometer is a visual indicator of a mutual-fund scheme's risk level. Check the latest scheme factsheet for the scheme-specific level.

**Source:** [SEBI Investor — Riskometer](https://investor.sebi.gov.in/riskometer.html)

### 8. Should I buy HDFC Flexi Cap Fund?
**Expected answer:** I can provide verified scheme facts, but I can't recommend buying, selling, holding or switching funds, or compare/predict returns. For neutral scheme information, consult the official [HDFC Mutual Fund factsheets](https://www.hdfcfund.com/mutual-funds/factsheets).

**Source:** [HDFC Mutual Fund — Factsheets](https://www.hdfcfund.com/mutual-funds/factsheets)

## Disclaimer snippet used in the UI

> **Facts-only. No investment advice.** This assistant provides factual information from public HDFC AMC, SEBI and AMFI sources only. It does not recommend buying, selling, holding or switching investments, provide portfolio/tax advice, or calculate, compare or predict returns. Do not enter PAN, Aadhaar, folio/account numbers, OTPs, email addresses or phone numbers.

## Technology stack

Python 3.10+, Streamlit, OpenAI API (`gpt-4o-mini`), OpenAI embeddings (`text-embedding-3-small`), LangChain and FAISS.

## Local setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set credentials in `.env` (never commit the file):

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Run with `streamlit run app.py`.

## Streamlit Community Cloud secrets

In Advanced settings → Secrets:

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-4o-mini"
```

## Tests

Run `pytest -q` and `python -m py_compile app.py`. The included policy tests cover common PII patterns and advice/performance refusal classification; they do not validate live crawling, factual correctness, citation entailment or OpenAI connectivity.

## Known limitations

- The source list is a candidate inventory, not proof every URL is live or ingested. Validate URLs and crawl results before submission.
- Some AMC pages may block automated requests, require JavaScript or change layout.
- PII detection uses basic regular expressions and is not complete data-loss prevention. Do not enter real personal information.
- Retrieval and citation correctness require live testing and human review.
- TER, SIP minimum, exit load and riskometer values can change; use the latest factsheet, SID, KIM or addendum and note its publication/as-of date.
- No return calculations, rankings, personalized recommendations, account access or transactions are implemented.
- This is an educational prototype, not a production financial-information service.
