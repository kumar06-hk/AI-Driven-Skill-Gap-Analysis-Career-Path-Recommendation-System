# AI-Driven Skill Gap Analysis & Career Path Recommendation System

Final Year Project (FYP) — a resume-to-job matching system that benchmarks multiple NLP approaches, identifies skill gaps, and recommends career progression and learning resources.

Multimedia University, Faculty of Computing and Informatics.

## Overview

- Benchmarked TF-IDF, DistilBERT, SBERT, and a hybrid TF-IDF + SBERT approach for resume-to-job retrieval
- Hybrid model (0.7 TF-IDF + 0.3 SBERT, using all-mpnet-base-v2) selected as the final matching model
- Rule-based resume skill extraction
- Skill gap analysis comparing resume skills against target job requirements
- Career path recommendation and learning resource recommendation from a Coursera dataset
- Five-page Streamlit dashboard: Dashboard, Resume Analysis, Job Matching, Recommendations, Career Path

## Key results

- **Hybrid model achieved 100% Top-10 query-label accuracy**, vs. TF-IDF (100%), SBERT (89.66%), DistilBERT (62.07%)
- Job matching results: 53.5% average match score, 80.0% best match, 46.5% average skill gap across Top-10 jobs
- 128 course recommendations and 147 job-specific recommendations generated
- 6,641 cleaned Coursera course records used for recommendation retrieval

## My contribution

Solo project — I designed and built the full pipeline: model benchmarking, resume skill extraction, skill gap analysis, career path and course recommendation logic, and the Streamlit dashboard.

## Tech stack

Python, Streamlit, Scikit-learn, Sentence-Transformers (SBERT), Hugging Face Transformers (DistilBERT), Pandas, NumPy

## Data

This project uses the following datasets, which are not included in this repo due to file size:

- **Coursera course dataset** — azrai99. Coursera Course Dataset [Data set]. Hugging Face. https://huggingface.co/datasets/azrai99/coursera-course-dataset
- **JobStreet job postings dataset** — Mahadan, A. JobStreet All Job Dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/azraimohamad/JobStreet-all-job-dataset
- **IT job postings dataset** — Papachristou, G. IT Job Post Descriptions [Data set]. Kaggle. https://www.kaggle.com/datasets/mscgeorges/itjobpostdescriptions

Download these and place them in a `data/` folder before running the notebooks/scripts.

## Files

- Notebooks — model benchmarking, skill gap analysis, career path recommendation, learning resource recommendation
- Streamlit app — the five-page dashboard (all notebook logic integrated into `Objective_6_dashboard.py`)
- `outputs/` — sample results generated while testing the notebooks individually (to verify each component worked before integration), not actual output from running the live dashboard. Includes recommendations, skill gap results, career roadmap, and a sample user profile (skills/education extracted from a resume, no personal identifiers)

## Note on resume data

Model comparison (Objective 1) was evaluated using synthetic resumes, included in this repo. The Streamlit dashboard is designed to accept a real resume uploaded by the end user at runtime — no real resumes are stored or included in this repo.

## Note on excluded files

Precomputed embeddings, TF-IDF matrices, cached similarity scores, and vectorizer/index files are not included since they're regenerable by running the pipeline on the source datasets.
