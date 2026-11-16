import pathlib

from matcher.core import find_skills, match, split_job, weak_bullets

EX = pathlib.Path(__file__).resolve().parents[1] / "examples"


def test_synonyms_map_to_one_skill():
    assert "scikit-learn" in find_skills("used sklearn for a model")
    assert "SQL" in find_skills("queried postgres")


def test_short_skill_names_need_word_boundaries():
    assert "R" not in find_skills("our team works remotely")
    assert "R" in find_skills("analysis in R and Python")
    assert "Java" not in find_skills("javascript only")


def test_split_job_sections():
    req, pref = split_job("Intro\nRequirements\n- Python\nNice to have\n- Docker\n")
    assert "Python" in req and "Docker" in pref and "Docker" not in req


def test_weak_bullets_are_ones_without_numbers():
    assert weak_bullets("- Built a model\n- Cut costs by 10%\n") == ["Built a model"]


def test_example_match():
    m = match((EX / "resume.txt").read_text(), (EX / "job.txt").read_text())
    assert "A/B testing" in m.required_missing
    assert {"Python", "SQL", "scikit-learn"} <= set(m.required_hit)
    assert "Tableau" in m.preferred_hit and "Docker" in m.preferred_missing
    assert 0 < m.score < 100


def test_perfect_match_scores_high():
    job = "Requirements\n- Python\n- SQL\n"
    assert match("I use Python and SQL daily.", job).score >= 85
