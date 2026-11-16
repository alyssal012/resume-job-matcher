"""Compare a resume to a job description.

Three signals:
1. Skill coverage: which skills the job asks for that the resume shows, split
   into required vs. preferred, using a taxonomy with synonyms ("sklearn" counts
   as scikit-learn).
2. Text similarity: TF-IDF cosine between the two documents.
3. Bullet quality: resume bullets with no number in them, since quantified
   impact is what recruiters scan for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS: dict[str, list[str]] = {
    "Python": ["python"], "SQL": ["sql", "postgres", "postgresql", "mysql", "sqlite"],
    "R": [r"\br\b"], "Java": [r"\bjava\b"], "JavaScript": ["javascript", "typescript", r"\bjs\b"],
    "pandas": ["pandas"], "NumPy": ["numpy"], "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "PyTorch": ["pytorch", "torch"], "TensorFlow": ["tensorflow", "keras"],
    "Machine learning": ["machine learning", r"\bml\b", "predictive model", "classification", "regression"],
    "Deep learning": ["deep learning", "neural network"],
    "LLMs": [r"\bllms?\b", "large language model", "gpt", "claude", "prompt engineering"],
    "RAG": [r"\brag\b", "retrieval-augmented", "retrieval augmented", "vector database", "embeddings"],
    "NLP": [r"\bnlp\b", "natural language processing", "text classification"],
    "Statistics": ["statistic", "hypothesis test", "t-test", "regression analysis", "inference"],
    "A/B testing": ["a/b test", "ab test", "experimentation", "experiment design"],
    "Data visualization": ["data visualization", "matplotlib", "seaborn", "plotly", "dashboards?"],
    "Tableau": ["tableau"], "Power BI": ["power bi"], "Excel": ["excel", "spreadsheet"],
    "APIs": [r"\bapis?\b", r"\brest\b", "fastapi", "flask"],
    "Git": [r"\bgit\b", "github"], "Docker": ["docker", "container"],
    "Cloud": [r"\baws\b", r"\bgcp\b", "azure", "google cloud"],
    "Spark": ["spark", "pyspark"], "Forecasting": ["forecast", "time series"],
    "Communication": ["communicat", "present", "stakeholder"],
}
REQUIRED_HEAD = re.compile(r"^\s*(requirements?|required|qualifications|minimum qualifications|what you.ll need|must have)\b", re.I | re.M)
PREFERRED_HEAD = re.compile(r"^\s*(preferred|nice to have|bonus|pluses|preferred qualifications)\b", re.I | re.M)


def find_skills(text: str) -> set[str]:
    t = text.lower()
    return {skill for skill, pats in SKILLS.items()
            if any(re.search(p if p.startswith(r"\b") else re.escape(p).replace(r"\?", "?"), t) for p in pats)}


def split_job(job: str) -> tuple[str, str]:
    """Return (required_text, preferred_text). If there are no headings, treat everything as required."""
    req, pref = REQUIRED_HEAD.search(job), PREFERRED_HEAD.search(job)
    if not req and not pref:
        return job, ""
    if req and pref:
        if req.start() < pref.start():
            return job[req.start():pref.start()], job[pref.start():]
        return job[req.start():], job[pref.start():req.start()]
    if pref:
        return job[:pref.start()], job[pref.start():]
    return job[req.start():], ""


def weak_bullets(resume: str) -> list[str]:
    bullets = [l.strip(" -•*\t") for l in resume.splitlines() if re.match(r"\s*[-•*]\s+", l)]
    return [b for b in bullets if not re.search(r"\d", b)]


@dataclass
class Match:
    score: float
    text_similarity: float
    required_hit: list[str] = field(default_factory=list)
    required_missing: list[str] = field(default_factory=list)
    preferred_hit: list[str] = field(default_factory=list)
    preferred_missing: list[str] = field(default_factory=list)
    unquantified_bullets: list[str] = field(default_factory=list)


def match(resume: str, job: str) -> Match:
    req_text, pref_text = split_job(job)
    have = find_skills(resume)
    req = find_skills(req_text)
    pref = find_skills(pref_text) - req
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    m = tfidf.fit_transform([resume, job])
    sim = float(cosine_similarity(m[0], m[1])[0, 0])

    req_cov = len(req & have) / len(req) if req else 1.0
    pref_cov = len(pref & have) / len(pref) if pref else 1.0
    # Required skills dominate; text similarity is a light tiebreaker.
    score = round(100 * (0.65 * req_cov + 0.20 * pref_cov + 0.15 * min(1.0, sim / 0.4)), 1)
    return Match(score, round(sim, 3),
                 sorted(req & have), sorted(req - have),
                 sorted(pref & have), sorted(pref - have),
                 weak_bullets(resume))
