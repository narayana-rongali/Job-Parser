# Job search + resume match + semi-automated apply helper
# Run: python job_agent.py   (needs Python 3.10+, see requirements.txt)

import os
import re
import math
import time
import webbrowser
from collections import Counter
import pandas as pd
from pypdf import PdfReader
from jobspy import scrape_jobs

# ---------------- CONFIG (edit these) ----------------
RESUME_PDF = "Narayana_Rongali_Resume.pdf"
TRACKER_FILE = "job_tracker.xlsx"

ROLES = [
    "Data Engineer",
    "Senior Data Engineer",
    "Azure Data Engineer",
    "Databricks Data Engineer",
    "PySpark Data Engineer",
]
LOCATIONS = ["Hyderabad, India", "Bengaluru, India", "Chennai, India"]
SITES = ["linkedin", "indeed", "naukri"]   # each site is tried separately
HOURS_OLD = 72                             # only jobs posted in the last 3 days
RESULTS_PER_SEARCH = 25

APPLY_THRESHOLD = 0.55     # score >= this  -> APPLY (resume is good as is)
TAILOR_THRESHOLD = 0.35    # score >= this  -> TAILOR (tweak resume first)
OPEN_TABS = True           # open best matches in your browser for one-click apply
MAX_TABS_PER_RUN = 10

# titles containing these words are dropped
TITLE_EXCLUDE = ["intern", "trainee", "director", "principal", "manager",
                 "architect", "head of", "vp ", "associate director"]
TITLE_MUST_HAVE = ["data engineer", "data engineering", "etl", "databricks",
                   "pyspark", "big data", "analytics engineer"]

# skills vocabulary used to compare JD vs resume
SKILL_VOCAB = [
    "python", "pyspark", "spark", "sql", "t-sql", "databricks", "delta lake",
    "azure", "azure data factory", "adf", "synapse", "adls", "azure devops",
    "key vault", "power bi", "etl", "elt", "data modeling", "data modelling",
    "star schema", "scd", "data warehouse", "data warehousing", "medallion",
    "airflow", "kafka", "snowflake", "dbt", "aws", "glue", "redshift", "s3",
    "gcp", "bigquery", "terraform", "docker", "kubernetes", "git", "ci/cd",
    "teradata", "mysql", "sql server", "unity catalog", "streaming",
    "data quality", "agile", "scrum", "pandas", "scala", "hadoop", "hive",
]
# ------------------------------------------------------

# ---- 1. read resume ----
reader = PdfReader(RESUME_PDF)
resume_text = ""
for page in reader.pages:
    resume_text = resume_text + " " + (page.extract_text() or "")
resume_text = resume_text.lower()
resume_skills = [s for s in SKILL_VOCAB if re.search(r"(?<![a-z])" + re.escape(s) + r"(?![a-z])", resume_text)]
print("Skills found in resume:", len(resume_skills))

# ---- 2. search jobs ----
frames = []
for role in ROLES:
    for location in LOCATIONS:
        for site in SITES:
            print("Searching:", role, "|", location, "|", site)
            try:
                df = scrape_jobs(
                    site_name=[site],
                    search_term=role,
                    location=location,
                    results_wanted=RESULTS_PER_SEARCH,
                    hours_old=HOURS_OLD,
                    country_indeed="India",
                )
                df["search_role"] = role
                df["search_location"] = location
                frames.append(df)
            except Exception as e:
                print("  skipped (", site, "):", str(e)[:120])
            time.sleep(3)  # be polite, avoid blocks

if len(frames) == 0:
    print("No jobs fetched. Try again later or reduce searches.")
    raise SystemExit

jobs = pd.concat(frames, ignore_index=True)
jobs = jobs.drop_duplicates(subset=["job_url"])
jobs["title"] = jobs["title"].fillna("")
jobs["description"] = jobs["description"].fillna("")
print("Total unique jobs fetched:", len(jobs))

# ---- 3. filter by title ----
title_lower = jobs["title"].str.lower()
keep = title_lower.apply(lambda t: any(k in t for k in TITLE_MUST_HAVE) and not any(x in t for x in TITLE_EXCLUDE))
jobs = jobs[keep].copy()
print("After title filter:", len(jobs))

# ---- 4. drop jobs already in tracker ----
old_tracker = pd.DataFrame()
if os.path.exists(TRACKER_FILE):
    old_tracker = pd.read_excel(TRACKER_FILE)
    jobs = jobs[~jobs["job_url"].isin(old_tracker["job_url"])].copy()
print("New jobs to score:", len(jobs))

if len(jobs) == 0:
    print("Nothing new today.")
    raise SystemExit

# ---- 5. match resume to each JD ----
jd_text_list = (jobs["title"] + " " + jobs["description"]).str.lower().tolist()

# TF-IDF + cosine similarity in plain Python (no scikit-learn / scipy needed)
stopwords = set("a an and are as at be by for from has have in is it of on or that the to with you your we our will can this their they who work working experience years year team strong good ability skills required preferred knowledge role job".split())
docs = [resume_text] + jd_text_list
doc_tokens = []
doc_freq = Counter()
for doc in docs:
    tokens = [t for t in re.findall(r"[a-z][a-z0-9\+#]+", doc) if t not in stopwords]
    doc_tokens.append(tokens)
    doc_freq.update(set(tokens))

n_docs = len(docs)
doc_vectors = []
for tokens in doc_tokens:
    counts = Counter(tokens)
    vec = {}
    for term, cnt in counts.items():
        idf = math.log((n_docs + 1) / (doc_freq[term] + 1)) + 1
        vec[term] = cnt * idf
    doc_vectors.append(vec)

resume_vec = doc_vectors[0]
resume_norm = math.sqrt(sum(v * v for v in resume_vec.values()))
tfidf_scores = []
for vec in doc_vectors[1:]:
    jd_norm = math.sqrt(sum(v * v for v in vec.values()))
    dot = 0.0
    for term, weight in vec.items():
        if term in resume_vec:
            dot = dot + weight * resume_vec[term]
    if jd_norm == 0 or resume_norm == 0:
        tfidf_scores.append(0.0)
    else:
        tfidf_scores.append(dot / (jd_norm * resume_norm))

final_scores = []
matched_list = []
missing_list = []
decisions = []
for i in range(len(jd_text_list)):
    jd = jd_text_list[i]
    jd_skills = [s for s in SKILL_VOCAB if re.search(r"(?<![a-z])" + re.escape(s) + r"(?![a-z])", jd)]
    matched = [s for s in jd_skills if s in resume_skills]
    missing = [s for s in jd_skills if s not in resume_skills]
    if len(jd_skills) > 0:
        skill_score = len(matched) / len(jd_skills)
    else:
        skill_score = 0.0
    tfidf_part = min(1.0, tfidf_scores[i] * 3)
    score = round(0.7 * skill_score + 0.3 * tfidf_part, 2)
    if len(jd) < 200:
        score = round(score * 0.8, 2)  # very short JD, be cautious
    if score >= APPLY_THRESHOLD:
        decision = "APPLY"
    elif score >= TAILOR_THRESHOLD:
        decision = "TAILOR"
    else:
        decision = "SKIP"
    final_scores.append(score)
    matched_list.append(", ".join(matched))
    missing_list.append(", ".join(missing))
    decisions.append(decision)

jobs["match_score"] = final_scores
jobs["decision"] = decisions
jobs["matched_skills"] = matched_list
jobs["missing_skills"] = missing_list
jobs["status"] = "New"
jobs["found_on"] = pd.Timestamp.now().strftime("%Y-%m-%d")

# ---- 6. save tracker ----
columns = ["found_on", "decision", "match_score", "title", "company", "location",
           "site", "date_posted", "job_url", "matched_skills", "missing_skills",
           "status", "search_role", "search_location"]
columns = [c for c in columns if c in jobs.columns]
jobs = jobs[columns].sort_values("match_score", ascending=False)
tracker = pd.concat([old_tracker, jobs], ignore_index=True)
tracker.to_excel(TRACKER_FILE, index=False)
print("Saved", len(jobs), "new jobs to", TRACKER_FILE)
print(jobs["decision"].value_counts().to_string())

# ---- 7. semi-automatic apply: open best matches in browser ----
if OPEN_TABS:
    to_open = jobs[jobs["decision"] == "APPLY"].head(MAX_TABS_PER_RUN)
    for idx, row in to_open.iterrows():
        print("Opening:", row["title"], "-", row["company"], "(", row["match_score"], ")")
        webbrowser.open(row["job_url"])
        tracker.loc[tracker["job_url"] == row["job_url"], "status"] = "Opened - review & submit"
        time.sleep(2)
    tracker.to_excel(TRACKER_FILE, index=False)

print("Done. Open", TRACKER_FILE, "- update the status column after you apply.")
