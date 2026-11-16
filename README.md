# Resume ↔ Job Matcher

A command-line tool that scores a resume against a job posting and tells you what to fix before you apply.

```
$ python -m matcher examples/resume.txt examples/job.txt
Match score: 65.4/100   (text similarity 0.08)

Required skills   ✓ Communication, Machine learning, Python, SQL, Statistics, scikit-learn
                  ✗ A/B testing
Preferred skills  ✓ Data visualization, Tableau
                  ✗ Cloud, Docker, LLMs, RAG

If you have these, say so explicitly. If not, they're the gap to close first:
  - A/B testing

4 bullet(s) have no numbers. Add scale or impact where you can:
  - Wrote SQL queries against a Postgres warehouse to automate the Monday sales report
  - Presented findings to regional managers
  ...
```

(The example resume and posting are fictional.)

## How it works

1. **Splits the posting** into *required* and *preferred* sections by their headings ("Requirements", "Nice to have", "Preferred qualifications" and so on). A missing required skill matters much more than a missing nice-to-have.
2. **Extracts skills** using a taxonomy of about 30 data/AI skills with synonyms, so "sklearn", "scikit learn" and "scikit-learn" all count as one skill, and "Postgres" counts as SQL. Short names like R and Java use word boundaries, so "remotely" doesn't count as R and "JavaScript" doesn't count as Java. Tests cover both.
3. **Scores** the match: 65% required-skill coverage, 20% preferred coverage, and 15% TF-IDF cosine similarity between the documents.
4. **Flags bullets without numbers**, because "built a model" reads much weaker than "built a model that cut overstock 12% across 40 stores."

### Why not just use embeddings?

Text similarity alone is a weak signal: in the example above, TF-IDF similarity is only 0.08 even though the candidate meets 6 of 7 requirements. Resumes and postings use different words for the same thing. An explicit skill taxonomy is less fancy but more accurate and fully explainable, so you can see exactly *why* you got your score. Embedding similarity would be a good addition as a fourth signal, not a replacement.

## Run it

```bash
pip install -r requirements.txt
python -m matcher path/to/resume.txt path/to/job.txt
pytest -q
```

To use it on a PDF resume, convert it to text first (for example `pdftotext resume.pdf resume.txt`).

## Ideas for next steps

- Batch mode: rank 50 saved postings by fit
- Learn skill synonyms from a corpus of postings instead of a hand-written list
- Suggest which projects on your resume best demonstrate each missing skill

## Stack

Python, scikit-learn (TF-IDF), regex, pytest
