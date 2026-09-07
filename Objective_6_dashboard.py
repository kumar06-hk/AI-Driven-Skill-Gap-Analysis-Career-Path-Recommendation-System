import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import hashlib
import re
import ast
import pickle
import time
import warnings
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.sparse import load_npz
import pdfplumber

warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FYP Career Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Color variables ── */
:root {
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --surface2:  #22263a;
    --border:    #2d3148;
    --accent:    #4f8ef7;
    --accent2:   #7c5cbf;
    --green:     #22c55e;
    --orange:    #f59e0b;
    --red:       #ef4444;
    --text:      #e8eaf6;
    --muted:     #8b91b0;
}

/* ── Top navbar ── */
.navbar {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d27 100%);
    border-bottom: 1px solid var(--border);
    padding: 1rem 2rem;
    margin: 0rem -4rem 2rem -4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.navbar-brand {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
}
.navbar-sub {
    font-size: 0.8rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.card-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── Metric row ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

/* ── Section header ── */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Skill badge ── */
.badge-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
.badge {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: var(--text);
    font-family: 'DM Mono', monospace;
}
.badge-green  { border-color: var(--green);  color: var(--green);  background: #052e16; }
.badge-red    { border-color: var(--red);    color: var(--red);    background: #1f0a0a; }
.badge-blue   { border-color: var(--accent); color: var(--accent); background: #0a1628; }
.badge-orange { border-color: var(--orange); color: var(--orange); background: #1c1008; }

/* ── Job card ── */
.job-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.job-card:hover { border-color: var(--accent); }
.job-title { font-size: 0.95rem; font-weight: 600; color: var(--text); }
.job-meta  { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Score bar ── */
.score-bar-bg {
    background: var(--surface2);
    border-radius: 999px;
    height: 6px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

/* ── Tier badge ── */
.tier-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.tier-junior   { background: #0a1628; color: var(--accent); border: 1px solid var(--accent); }
.tier-mid      { background: #1c1008; color: var(--orange); border: 1px solid var(--orange); }
.tier-senior   { background: #052e16; color: var(--green);  border: 1px solid var(--green); }
.tier-manager  { background: #1f0a0a; color: var(--red);    border: 1px solid var(--red); }

/* ── Roadmap step ── */
.roadmap-step {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    position: relative;
}
.roadmap-step.active { border-color: var(--accent); background: #0d1829; }
.roadmap-step.done   { border-color: var(--green);  background: #040f07; }
.roadmap-label { font-size: 0.7rem; color: var(--muted); font-family: 'DM Mono', monospace; }
.roadmap-tier  { font-size: 1rem; font-weight: 700; color: var(--text); margin: 0.2rem 0; }

/* ── Course card ── */
.course-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.course-name { font-size: 0.88rem; font-weight: 600; color: var(--text); }
.course-meta { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Upload zone ── */
.upload-zone {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    color: var(--muted);
}
.upload-zone h3 { color: var(--text); font-size: 1.1rem; }

/* ── Status pill ── */
.status-ok  { color: var(--green);  font-size: 0.8rem; }
.status-err { color: var(--red);    font-size: 0.8rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    padding: 0.5rem 1.5rem;
    width: 100%;
}
.stButton > button:hover { background: #3a7ae8; }
div[data-testid="stFileUploader"] { margin-top: 0.5rem; }
.stProgress > div > div { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
DATA_DIR  = "data"
SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
JOB_MATCH_LIMIT = 10
RECOMMENDATION_SKILL_LIMIT = 10
TIER_ORDER = ['Junior', 'Mid-Level', 'Senior', 'Manager/Lead']
TIER_CLASS = {
    'Junior': 'tier-junior',
    'Mid-Level': 'tier-mid',
    'Senior': 'tier-senior',
    'Manager/Lead': 'tier-manager'
}

SKILL_KEYWORDS = [
    'python','java','javascript','typescript','c++','c#','c','r','php','ruby',
    'go','swift','kotlin','scala','matlab','perl','bash','shell','vba',
    'html','css','react','angular','vue','jquery','bootstrap','node.js',
    'django','flask','fastapi','spring','express','.net','laravel',
    'machine learning','deep learning','nlp','natural language processing',
    'computer vision','data science','data analysis','data mining','statistics',
    'statistical analysis','feature engineering','etl',
    'pandas','numpy','scikit-learn','tensorflow','pytorch','keras','xgboost',
    'matplotlib','seaborn','plotly','opencv',
    'sql','mysql','postgresql','mongodb','redis','oracle','sqlite',
    'elasticsearch','nosql','ms sql','dynamodb',
    'aws','azure','gcp','docker','kubernetes','terraform','ansible',
    'jenkins','git','ci/cd','linux','unix','airflow','kafka','spark',
    'hadoop','databricks','snowflake',
    'tableau','power bi','excel','sas','spss','google analytics',
    'microsoft office','microsoft office suite','microsoft excel','ibm db2','powerpoint',
    'agile','scrum','jira','project management','devops','uml',
    'accounting','bookkeeping','auditing','taxation','financial reporting',
    'financial analysis','budgeting','forecasting','sap','quickbooks','xero',
    'marketing','digital marketing','seo','sem','social media','social media marketing',
    'social media management','content marketing','content writing','copywriting',
    'email marketing','google ads','google analytics','crm','salesforce','market research',
    'campaign management','brand awareness','customer engagement','canva',
    'recruitment','payroll','hris','training','performance management',
    'microsoft 365','bank reconciliation','cash reconciliation','fixed asset','fixed asset schedules',
    'statutory submission','statutory submissions','audit','payroll','million accounting system','auto count accounting system',
    'autocad','solidworks','electrical engineering','mechanical engineering',
    'civil engineering','quality control','iso',
    'patient care','clinical','nursing','pharmacy','laboratory',
    'communication','presentation','leadership','teamwork','negotiation',
    'customer service','problem solving','time management'
]

IGNORE_GAP_SKILLS = {'communication', 'training'}

# Extra domain keywords used when comparing resume skills with job requirements.
for _extra_skill in [
    'seo','sem','social media marketing','social media management','content writing','content marketing',
    'copywriting','canva','campaign management','market research','brand awareness','customer engagement',
    'email marketing','google ads','bank reconciliation','cash reconciliation','fixed asset schedules',
    'statutory submissions','million accounting system','auto count accounting system','microsoft 365',
    'financial data management','analytical thinking','adaptability',
    'jupyter notebook','vs code','codeblock','business analysis','business analytics',
    'data visualization','requirement gathering','requirements gathering','reporting',
    'employee relations','hr administration','training coordination','interview','conflict resolution',
    'accounts payable','tax documentation'
]:
    if _extra_skill not in SKILL_KEYWORDS:
        SKILL_KEYWORDS.append(_extra_skill)

SKILL_ALIASES = {
    'sql': {'sql', 'mysql', 'postgresql', 'sqlite', 'oracle', 'ms sql', 'ibm db2'},
    'mysql': {'mysql'},
    'java': {'java'},
    'javascript': {'javascript', 'js'},
    'c': {'c'},
    'c++': {'c++', 'cpp'},
    'c#': {'c#', 'c sharp'},
    '.net': {'.net', 'dotnet', 'asp.net'},
    'power bi': {'power bi', 'powerbi'},
    'microsoft office': {'microsoft office', 'office suite', 'ms office'},
    'microsoft 365': {'microsoft 365', 'microsoft office', 'office suite'},
    'audit': {'audit', 'auditing'},
    'social media marketing': {'social media marketing', 'social media management', 'social media'},
}

def skill_match(required_skill, user_skills):
    req = str(required_skill).lower().strip()
    user = {str(s).lower().strip() for s in user_skills}
    if req in user:
        return True
    aliases = SKILL_ALIASES.get(req, {req})
    for u in user:
        if u in aliases:
            return True
        if req not in {'c', 'java'} and u not in {'c', 'java'}:
            if len(req) >= 4 and len(u) >= 4 and (req in u or u in req):
                return True
    return False

def clean_gap_skills(skills):
    return sorted([s for s in skills if str(s).lower().strip() not in IGNORE_GAP_SKILLS])

def valid_url(url):
    url = str(url).strip()
    if not url or url.lower() in {'nan', 'none', 'n/a', 'na'}:
        return ''
    if url.startswith('www.'):
        url = 'https://' + url
    if not url.startswith(('http://', 'https://')):
        return ''
    return url


STOP_HEADERS = [
    'work experience','experience','employment','professional experience',
    'career history','work history','education','academic','qualifications',
    'academic background','professional summary','summary','profile','about me','objective',
    'projects','project','academic projects','personal projects',
    'certifications','certification','licenses','awards','languages',
    'interests','hobbies','references','volunteer','achievements',
    'publications','extracurricular','skills & achievements','skill & achievements',
    'technical skills','soft skills','skills summary','key skills'
]
SKILLS_HEADERS = [
    'skills','skills summary','technical skills','key skills','competencies',
    'tools','technologies','core competencies','skill set','technical expertise',
    'my toolkit','toolkit','tech stack','areas of expertise','professional skills',
    'it skills','software skills','programming skills','programming languages',
    'languages & tools','languages and tools','tools & technologies',
    'tools and technologies','technical proficiencies','proficiencies',
    'capabilities','technical summary','relevant skills','hard skills',
    'soft skills','skills & achievements','skill & achievements','key competencies'
]
GENERIC_PHRASES = {
    'documentation','problem solving','communication','teamwork','collaboration',
    'ability','skills','expertise','experience','knowledge','understanding',
    'developing','develop','including','sector','service','services','models',
    'model','foundation','foundations','required','related','education',
    'research','performance','safety','quality','technical','architecture',
    'web','applications','summary','objective','profile','history',
    'career','professional'
}
SKILL_PREFIXES = re.compile(
    r'^(basic knowledge of|intermediate knowledge of|advanced knowledge of|'
    r'basic level knowledge of|intermediate level knowledge of|'
    r'advanced level knowledge of|knowledge of|knowledge in|'
    r'proficient in|proficient with|experience in|experience with|'
    r'familiar with|familiarity with|exposure to|understanding of|'
    r'working knowledge of|hands.on experience with|ability to use|'
    r'skilled in|expertise in)\s+', re.IGNORECASE
)
SKILL_SUFFIX = re.compile(r'\s*\(.*?\)\s*$')
MONTH_MAP = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
}

# ── Resume parsing helpers ────────────────────────────────────
def _clean(t):
    t = str(t).replace('\r', '\n')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{2,}', '\n', t)
    t = re.sub(r'\(cid:\d+\)', ' ', t)
    return t.strip()

def read_pdf(path):
    standard_pages, column_pages = [], []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            std = page.extract_text() or ''
            standard_pages.append(std)
            x0, top, x1, bottom = page.bbox
            w = x1 - x0
            mid = x0 + w * 0.50
            try:
                left  = page.crop((x0, top, mid + w * 0.02, bottom))
                right = page.crop((mid - w * 0.02, top, x1, bottom))
                col   = (left.extract_text() or '') + '\n' + (right.extract_text() or '')
            except Exception:
                col = std
            column_pages.append(col)
    std_full = _clean('\n'.join(standard_pages))
    col_full = _clean('\n'.join(column_pages))

    # Keep both normal and column-wise extraction.
    # This helps resumes that have side-by-side sections or two-column layouts.
    # Duplicates are removed later when skills, education and roles are cleaned.
    combined = _clean(std_full + '\n' + col_full)
    return combined

def normalize_line(s):
    return re.sub(r'\s+', ' ', s.strip())

def looks_like_header(line):
    l = normalize_line(line)
    if not l or len(l) > 60:
        return False
    letters = re.sub(r'[^A-Za-z]+', '', l)
    if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.75:
        return True
    # Short title-style lines without sentence punctuation can be section headers.
    if not re.search(r'[.!?]', l) and len(l.split()) <= 6:
        return True
    return False

def _clean_header_text(line):
    cleaned = normalize_line(line).lower()
    cleaned = re.sub(r'^[^a-z]+|[^a-z]+$', '', cleaned)
    cleaned = re.sub(r'[^a-z&/ ]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def _matches_section_header(line, header_keywords):
    # Match actual section headings only. This prevents words like
    # "experienced" inside a summary sentence from being treated as a section.
    h = _clean_header_text(line)
    if not h or len(h.split()) > 6:
        return False
    for k in header_keywords:
        k = _clean_header_text(k)
        if h == k:
            return True
        if h.startswith(k + ' ') and len(h.split()) <= len(k.split()) + 2:
            return True
    return False

def find_section_block(text, header_keywords, max_lines=120):
    lines = [normalize_line(x) for x in text.split('\n') if normalize_line(x)]
    start_idx = None
    for i, ln in enumerate(lines):
        if _matches_section_header(ln, header_keywords):
            start_idx = i
            break
    if start_idx is None:
        return ''
    end_idx = min(len(lines), start_idx + max_lines)
    for j in range(start_idx + 1, end_idx):
        if _matches_section_header(lines[j], STOP_HEADERS):
            end_idx = j
            break
    return '\n'.join(lines[start_idx + 1:end_idx]).strip()


def find_all_section_blocks(text, header_keywords, max_lines=120):
    lines = [normalize_line(x) for x in text.split('\n') if normalize_line(x)]
    blocks = []
    for i, ln in enumerate(lines):
        if not _matches_section_header(ln, header_keywords):
            continue
        end_idx = min(len(lines), i + max_lines)
        for j in range(i + 1, end_idx):
            # For skill extraction, allow sub-headers such as Technical Skills and Soft Skills.
            # Do not stop on real skills such as Project Management, because some skills look like headings.
            if _matches_section_header(lines[j], STOP_HEADERS) and not _matches_section_header(lines[j], header_keywords):
                if scan_known_skills(lines[j]):
                    continue
                end_idx = j
                break
        block = '\n'.join(lines[i + 1:end_idx]).strip()
        if block:
            blocks.append(block)
    return blocks

def split_skills_from_line(line):
    line = str(line).strip()
    if not line:
        return []
    line = re.sub(r'^[•\-\*\u2022\d\.\)]+\s*', '', line).strip()
    line = line.replace('MYSQL(Basic)', 'MYSQL')

    # Keep only the value after a normal skill label.
    # Example: Programming Languages: Python, Java
    if ':' in line and len(line.split(':', 1)[0]) < 35:
        label = line.split(':', 1)[0].lower()
        if any(k in label for k in ['skill', 'language', 'tool', 'database', 'programming', 'development', 'software', 'analysis', 'analytics', 'visualization']):
            line = line.split(':', 1)[1].strip()

    # Remove level wording but keep the actual skills.
    line = re.sub(r'\b(basic|intermediate|advanced)\s+(level\s+)?knowledge\s+of\s+', '', line, flags=re.I)
    line = re.sub(r'\bknowledge\s+of\s+', '', line, flags=re.I)
    line = re.sub(r'\bproficient\s+(in|with)\s+', '', line, flags=re.I)

    parts = re.split(r'[,;/|•\n]+|\s+and\s+', line)
    cleaned = []
    for p in parts:
        p = p.strip().strip(' .;:-"')
        if not p:
            continue
        low = p.lower()
        if re.search(r'@|http|www\.|\(cid:', low):
            continue
        if re.fullmatch(r'[\d\W]+', p):
            continue
        if len(p) < 2 and p.lower() not in {'r', 'c'}:
            continue
        if len(p) > 70 or len(p.split()) > 6:
            continue
        cleaned.append(p)
    return cleaned


def clean_skill_name(s):
    s = str(s).strip()
    s = SKILL_SUFFIX.sub('', s).strip()
    s = SKILL_PREFIXES.sub('', s).strip()
    s = re.sub(r'\s+', ' ', s).strip(' .;:-"')
    return s


# These words should not become skills even if they appear in the skill section.
LANGUAGE_CERT_WORDS = {
    'english', 'malay', 'tamil', 'mandarin', 'chinese', 'bahasa malaysia',
    'muet', 'certificate', 'certification', 'band', 'cgpa', 'lcci', 'dean', 'dean’s list',
    'sololearn', 'level 2', 'bahasa melayu'
}

# Controlled vocabulary used by the live dashboard.
# I added more domain terms so the prototype can work with ICT, accounting and marketing resumes.
CONTROLLED_SKILLS = {
    # ICT / data
    'python': 'Python', 'java': 'Java', 'javascript': 'JavaScript', 'typescript': 'TypeScript',
    'c++': 'C++', 'cpp': 'C++', 'c#': 'C#', 'c sharp': 'C#', 'c': 'C', 'r': 'R',
    'php': 'PHP', 'ruby': 'Ruby', 'go': 'Go', 'kotlin': 'Kotlin', 'scala': 'Scala',
    'html': 'HTML', 'css': 'CSS', 'sql': 'SQL', 'mysql': 'MySQL', 'ms sql': 'MS SQL',
    'ibm db2': 'IBM DB2', 'db2': 'IBM DB2', 'oracle': 'Oracle', 'postgresql': 'PostgreSQL',
    'mongodb': 'MongoDB', 'nosql': 'NoSQL', 'django': 'Django', 'flask': 'Flask',
    'react': 'React', 'angular': 'Angular', 'vue': 'Vue', 'node.js': 'Node.js', 'node js': 'Node.js',
    'data analysis': 'Data Analysis', 'data analytics': 'Data Analysis', 'data science': 'Data Science',
    'machine learning': 'Machine Learning', 'deep learning': 'Deep Learning',
    'statistics': 'Statistics', 'statistical analysis': 'Statistical Analysis',
    'nlp': 'NLP', 'natural language processing': 'Natural Language Processing',
    'computer vision': 'Computer Vision', 'power bi': 'Power BI', 'tableau': 'Tableau',
    'microsoft excel': 'Microsoft Excel', 'excel': 'Microsoft Excel',
    'microsoft office suite': 'Microsoft Office Suite', 'microsoft office': 'Microsoft Office Suite',
    'ms office': 'Microsoft Office Suite', 'office suite': 'Microsoft Office Suite',
    'aws': 'AWS', 'azure': 'Azure', 'gcp': 'GCP', 'docker': 'Docker', 'kubernetes': 'Kubernetes',
    'devops': 'DevOps', 'ci/cd': 'CI/CD', 'linux': 'Linux', 'git': 'Git',
    'project management': 'Project Management', 'problem solving': 'Problem Solving',
    'analytical thinking': 'Analytical Thinking', 'time management': 'Time Management',
    'communication': 'Communication', 'adaptability': 'Adaptability', 'teamwork': 'Teamwork',
    'leadership': 'Leadership', 'presentation': 'Presentation',

    # accounting / finance
    'accounting': 'Accounting', 'bookkeeping': 'Bookkeeping', 'book-keeping': 'Bookkeeping',
    'financial reporting': 'Financial Reporting', 'financial analysis': 'Financial Analysis',
    'financial data management': 'Financial Data Management', 'bank reconciliation': 'Bank Reconciliation',
    'bank reconciliations': 'Bank Reconciliation', 'cash reconciliation': 'Cash Reconciliation',
    'audit': 'Audit', 'auditing': 'Auditing', 'payroll': 'Payroll', 'taxation': 'Taxation',
    'budgeting': 'Budgeting', 'forecasting': 'Forecasting', 'fixed asset': 'Fixed Asset',
    'fixed asset schedules': 'Fixed Asset Schedules', 'statutory submissions': 'Statutory Submissions',
    'statutory submission': 'Statutory Submissions', 'million accounting system': 'Million Accounting System',
    'auto count accounting system': 'Auto Count Accounting System', 'autocount': 'Auto Count Accounting System',
    'microsoft 365': 'Microsoft 365', 'xero': 'Xero', 'quickbooks': 'QuickBooks', 'sap': 'SAP',

    # marketing / business
    'digital marketing': 'Digital Marketing', 'seo': 'SEO', 'sem': 'SEM',
    'social media marketing': 'Social Media Marketing', 'social media management': 'Social Media Management',
    'social media': 'Social Media', 'content writing': 'Content Writing', 'content marketing': 'Content Marketing',
    'copywriting': 'Copywriting', 'google analytics': 'Google Analytics', 'canva': 'Canva',
    'campaign management': 'Campaign Management', 'market research': 'Market Research',
    'brand awareness': 'Brand Awareness', 'customer engagement': 'Customer Engagement',
    'email marketing': 'Email Marketing', 'google ads': 'Google Ads', 'crm': 'CRM',
    'customer service': 'Customer Service', 'sales': 'Sales',

    # HR / business / general office
    'business analysis': 'Business Analysis', 'business analytics': 'Business Analytics',
    'data visualization': 'Data Visualization', 'requirement gathering': 'Requirement Gathering',
    'requirements gathering': 'Requirement Gathering', 'reporting': 'Reporting',
    'data entry': 'Data Entry', 'employee relations': 'Employee Relations',
    'hr administration': 'HR Administration', 'training coordination': 'Training Coordination',
    'interview': 'Interview', 'interviewing': 'Interview', 'conflict resolution': 'Conflict Resolution',
    'critical thinking': 'Critical Thinking', 'team management': 'Team Management',
    'accounts payable': 'Accounts Payable', 'tax documentation': 'Tax Documentation',
    'recruitment': 'Recruitment', 'performance management': 'Performance Management',
    'hris': 'HRIS', 'negotiation': 'Negotiation',
    'jupyter notebook': 'Jupyter Notebook', 'vs code': 'VS Code', 'codeblock': 'Codeblock'
}

# Longer phrases must be matched before short words such as C and R.
CONTROLLED_SKILL_ITEMS = sorted(CONTROLLED_SKILLS.items(), key=lambda x: len(x[0]), reverse=True)


def _normal_text_for_skill_scan(text):
    t = str(text).lower()
    t = t.replace('mysql(basic)', 'mysql')
    t = re.sub(r'(?<=[a-z])[-–—](?=[a-z])', ' ', t)
    t = re.sub(r'[^a-z0-9+#./&\- ]+', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return ' ' + t + ' '


def is_valid_extracted_skill(s):
    s = clean_skill_name(s)
    low = s.lower().strip()
    if not low or len(low) > 70:
        return False
    if len(low) < 2 and low not in {'r', 'c'}:
        return False
    if re.search(r'@|http|www\.|\+?\d{6,}|\b20\d{2}\b|\b19\d{2}\b', low):
        return False
    if any(x in low for x in ['fluent in', 'native in', 'spoken', 'written', 'linkedin']):
        return False
    if low in LANGUAGE_CERT_WORDS:
        return False
    if any(x in low for x in [' certificate', ' certification', ' band ', ' cgpa', 'dean']):
        return False
    if len(low.split()) > 6:
        return False
    return True


def scan_known_skills(text):
    t = _normal_text_for_skill_scan(text)
    found = []
    taken_spans = []
    for key, pretty in CONTROLLED_SKILL_ITEMS:
        # Use strict boundary matching so single letters like C/R do not match inside normal words.
        pattern = r'(?<![a-z0-9+#])' + re.escape(key) + r'(?![a-z0-9+#])'
        if re.search(pattern, t):
            found.append(pretty)
    return found


def match_controlled_skill(raw_skill):
    low = clean_skill_name(raw_skill).lower().strip()
    if low in CONTROLLED_SKILLS:
        return CONTROLLED_SKILLS[low]
    for key, pretty in CONTROLLED_SKILL_ITEMS:
        pretty_low = pretty.lower()
        if low == pretty_low:
            return pretty
        # Allow phrase containment for longer skill names only.
        # This avoids mistakes such as SQL becoming PostgreSQL or Java becoming JavaScript.
        if len(key) >= 4 and re.search(r'(?<![a-z0-9+#])' + re.escape(key) + r'(?![a-z0-9+#])', low):
            return pretty
    return None


def _skills_from_declared_section(text):
    blocks = find_all_section_blocks(text, SKILLS_HEADERS, max_lines=120)
    if not blocks:
        return []

    out = []
    for block in blocks:
        for ln in block.split('\n'):
            raw_ln = ln.strip()
            if not raw_ln:
                continue
            low_ln = raw_ln.lower()
            # Skip non-skill sections only when the line itself starts with them.
            # Do not skip "Programming Languages: Python, Java" because that is a real skill label.
            if re.match(r'^(languages?|certifications?|certification|reference)\b', low_ln):
                continue
            if any(x in low_ln for x in ['dean\'s list', 'dean’s list']):
                continue
            # Skip plain section labels, but keep short lines such as "Java, SQL"
            # because two-column PDFs sometimes split skill lists across lines.
            line_hits = scan_known_skills(raw_ln)
            if ':' not in raw_ln and looks_like_header(raw_ln) and len(raw_ln.split()) <= 4 and not line_hits:
                continue

            # First scan the full line for controlled skills. This handles lines such as
            # "Programming Languages: Python, C++, Java, SQL" and side-by-side PDF text.
            out.extend(line_hits)

            # Then split list-like lines and keep only valid controlled skills.
            for raw in split_skills_from_line(raw_ln):
                if not is_valid_extracted_skill(raw):
                    continue
                canonical = match_controlled_skill(raw)
                if canonical:
                    out.append(canonical)
    return out

def dedupe_skills(skills):
    seen, out = set(), []
    for s in skills:
        s = clean_skill_name(s)
        if not is_valid_extracted_skill(s):
            continue
        low = s.lower()
        if low not in seen:
            seen.add(low)
            out.append(s)

    # Remove weaker duplicates when a clearer version exists.
    lows = [x.lower() for x in out]
    remove_if_present = {
        'excel': 'microsoft excel',
        'microsoft office': 'microsoft office suite',
        'social media': 'social media marketing',
        'audit': 'auditing',
        'fixed asset': 'fixed asset schedules',
        'statutory submission': 'statutory submissions',
        'db2': 'ibm db2'
    }
    final = []
    for s in out:
        low = s.lower()
        stronger = remove_if_present.get(low)
        if stronger and stronger in lows:
            continue
        final.append(s)
    return final


def extract_skills(text, max_skills=40):
    # Prefer the declared Skills / Technical Skills sections.
    # If a resume uses a side-by-side layout, read_pdf keeps both normal and column-wise text,
    # and this function checks all skill sections found in the combined text.
    section_skills = _skills_from_declared_section(text)

    if section_skills:
        merged = dedupe_skills(section_skills)
        # Add a small number of clean domain clues from summary/experience only when they are not already in the skills section.
        # This supports resumes where important tools are mentioned in experience, without letting project sentences become skills.
        if len(merged) < 8:
            summary_block = find_section_block(text, ['professional summary', 'summary', 'profile', 'about me'], 40)
            exp_block = find_section_block(text, ['work experience', 'experience', 'employment', 'professional experience'], 80)
            extra = scan_known_skills(summary_block + '\n' + exp_block)
            merged = dedupe_skills(merged + extra)
        return merged[:max_skills]

    # Fallback only for resumes without a clear Skills section.
    fallback = scan_known_skills(text)
    return dedupe_skills(fallback)[:max_skills]


def extract_education(text, max_items=6):
    DEGREE_WORDS = ['bachelor','master','phd','ph.d','diploma','degree','foundation',
                    'undergraduate','bsc','msc','mba','honours','hons','associate','certificate']
    block = find_section_block(text, ['education','academic background','qualifications'], 120)
    if not block: return []
    lines = [x.strip() for x in block.split('\n') if x.strip()]
    out, seen = [], set()
    for ln in lines:
        if any(d in ln.lower() for d in DEGREE_WORDS):
            if ln.lower() not in seen:
                seen.add(ln.lower())
                out.append(ln[:160])
    return out[:max_items]

def extract_experience_roles(text, max_roles=8):
    ROLE_HINTS = ['engineer','developer','analyst','manager','consultant','officer',
                  'assistant','specialist','designer','architect','intern','scientist',
                  'researcher','lead','head','director','coordinator','technician',
                  'nurse','doctor','accountant','teacher','lecturer','executive',
                  'cashier','storekeeper']
    block = find_section_block(
        text, ['work experience','experience','employment','professional experience'], 180)
    if not block: return []
    lines = [x.strip() for x in block.split('\n') if x.strip()]
    roles, seen = [], set()
    for ln in lines:
        low = ln.lower().strip()
        if len(ln) > 100 or ln.endswith('.'):
            continue
        if any(low.startswith(x) for x in ['managed ', 'performed ', 'handled ', 'processed ', 'assisted ', 'while ', 'and ', 'ensured ', 'prepared ', 'supported ', 'maintained ']):
            continue
        has_role_word = any(h in low for h in ROLE_HINTS)
        looks_title = ('|' in ln) or (ln.isupper() and len(ln.split()) <= 5) or has_role_word
        if has_role_word and looks_title:
            if low not in seen:
                seen.add(low)
                roles.append(ln)
    return roles[:max_roles]

def _date_to_month_index(dt):
    return dt.year * 12 + dt.month

def _month_index_to_pair(idx):
    y = idx // 12
    m = idx % 12
    if m == 0:
        y -= 1
        m = 12
    return y, m

def extract_years_experience(text):
    # Estimate experience from real work-experience date ranges only.
    # It ignores education and project dates to avoid inflated years.
    block = find_section_block(
        text,
        ['work experience', 'experience', 'employment', 'professional experience'],
        180
    )
    if not block:
        return 0.0

    t = block.lower().replace('–', '-').replace('—', '-')
    now = datetime.now()
    periods = []

    month_map = MONTH_MAP
    month_name = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*'

    # Example: July 2025 - January 2026, Mar 2022 - Present
    month_pattern = re.compile(
        month_name + r'\s+(20\d{2}|19\d{2})\s*-\s*'
        r'(present|current|now|' + month_name + r'\s+(20\d{2}|19\d{2}))',
        re.I
    )
    for m in month_pattern.finditer(t):
        sm = m.group(1)[:3].lower()
        sy = int(m.group(2))
        end_text = m.group(3).strip().lower()
        start_dt = datetime(sy, month_map.get(sm, 1), 1)
        if end_text in {'present', 'current', 'now'}:
            end_dt = now
        else:
            em = m.group(4)[:3].lower()
            ey = int(m.group(5))
            end_dt = datetime(ey, month_map.get(em, 1), 1)
        if end_dt >= start_dt:
            periods.append((start_dt, end_dt, 'month'))

    # Remove month ranges before year-only search so we do not double count the same line.
    t_no_month_ranges = month_pattern.sub(' ', t)

    # Example: 2021 - 2022, 2022 - Present.
    # For year-only completed ranges, count end-start years, not both full years.
    year_pattern = re.compile(r'\b(20\d{2}|19\d{2})\s*-\s*(present|current|now|20\d{2}|19\d{2})\b', re.I)
    for m in year_pattern.finditer(t_no_month_ranges):
        sy = int(m.group(1))
        ey_text = m.group(2).lower()
        if ey_text in {'present', 'current', 'now'}:
            start_dt = datetime(sy, 1, 1)
            end_dt = now
        else:
            ey = int(ey_text)
            if ey < sy:
                continue
            # 2021-2022 is treated as about 1 year because month is not provided.
            start_dt = datetime(sy, 1, 1)
            end_dt = datetime(ey, 1, 1)
        if end_dt >= start_dt:
            periods.append((start_dt, end_dt, 'year'))

    if not periods:
        return 0.0

    months = set()
    for start_dt, end_dt, kind in periods:
        start_idx = _date_to_month_index(datetime(start_dt.year, start_dt.month, 1))
        end_idx = _date_to_month_index(datetime(end_dt.year, end_dt.month, 1))
        # Use exclusive end for year-only completed ranges so 2021-2022 becomes 12 months.
        if kind == 'year' and end_dt.month == 1 and end_dt.day == 1 and end_dt.year != now.year:
            end_idx -= 1
        for idx in range(start_idx, end_idx + 1):
            months.add(idx)

    return round(len(months) / 12, 1)

# ── Data loaders (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sbert():
    return SentenceTransformer(SBERT_NAME)

@st.cache_data(show_spinner=False)
def load_jobs():
    clean_path = os.path.join(DATA_DIR, 'jobstreet_cleaned.csv')
    raw_path   = os.path.join(DATA_DIR, 'jobstreet_all_job_dataset.csv')
    path = clean_path if os.path.exists(clean_path) else raw_path
    df = pd.read_csv(path)
    def detect(candidates):
        for c in candidates:
            if c in df.columns: return c
        return None
    cols = {
        'title':   detect(['job_title','Job Title','title','position']),
        'desc':    detect(['job_description','descriptions','description','desc','requirements']),
        'cat':     detect(['category','Category','field','industry','job_category','specialization']),
        'company': detect(['company','Company','company_name','employer']),
        'loc':     detect(['location','Location','city','state']),
    }
    return df, cols

@st.cache_resource(show_spinner=False)
def load_job_index():
    vec_path  = os.path.join(DATA_DIR, 'job_tfidf_vec.pkl')
    mat_path  = os.path.join(DATA_DIR, 'job_tfidf_matrix.npz')
    emb_path  = os.path.join(DATA_DIR, 'job_embs.npy')
    old_vec   = os.path.join(DATA_DIR, 'tfidf_vec.pkl')

    tfidf_vec = None
    job_tfidf = None
    job_embs  = None

    if os.path.exists(vec_path):
        with open(vec_path, 'rb') as f:
            tfidf_vec = pickle.load(f)
    elif os.path.exists(old_vec):
        with open(old_vec, 'rb') as f:
            tfidf_vec = pickle.load(f)

    if os.path.exists(mat_path):
        job_tfidf = load_npz(mat_path)

    if os.path.exists(emb_path):
        job_embs = np.load(emb_path)

    return tfidf_vec, job_tfidf, job_embs

@st.cache_data(show_spinner=False)
def load_courses():
    path = os.path.join(DATA_DIR, 'coursera_cleaned.csv')
    if not os.path.exists(path):
        raise FileNotFoundError('coursera_cleaned.csv not found. Run Objective 4 first.')

    df = pd.read_csv(path)

    def detect(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    cols = {
        'name':   detect(['Title','Course Name','name','title','course_name']),
        'desc':   detect(['course_description','Course Description','description','Description','desc']),
        'skills': detect(['Skills','skills','skill_tags','tags','topics']),
        'org':    detect(['Organization','University','organization','org','provider']),
        'rating': detect(['Ratings','Course Rating','Rating','rating','stars']),
        'diff':   detect(['Difficulty','Difficulty Level','Level','level','difficulty']),
        'url':    detect(['course_url','Course URL','url','URL','link','course_link']),
    }

    for col in ['rating_norm', 'rating_display', 'course_text']:
        if col not in df.columns:
            if col == 'rating_norm':
                if cols['rating']:
                    df[cols['rating']] = pd.to_numeric(df[cols['rating']], errors='coerce')
                    max_r = df[cols['rating']].max()
                    df['rating_norm'] = df[cols['rating']].fillna(df[cols['rating']].mean()) / max_r if pd.notna(max_r) and max_r > 0 else 0.5
                else:
                    df['rating_norm'] = 0.5
            elif col == 'rating_display':
                if cols['rating']:
                    ratings = pd.to_numeric(df[cols['rating']], errors='coerce')
                    df['rating_display'] = ratings.apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'Not rated')
                else:
                    df['rating_display'] = 'Not rated'
            elif col == 'course_text':
                def build_text(row):
                    parts = []
                    for k in ['name','skills','desc']:
                        if cols[k]:
                            parts.append(str(row.get(cols[k],''))[:500])
                    return ' '.join(parts)
                df['course_text'] = df.apply(build_text, axis=1)

    return df, cols

@st.cache_resource(show_spinner=False)
def load_course_index(courses_df):
    vec_path = os.path.join(DATA_DIR, 'course_tfidf_vec.pkl')
    mat_path = os.path.join(DATA_DIR, 'course_tfidf_matrix.npz')
    emb_path = os.path.join(DATA_DIR, 'course_embs.npy')

    course_vec = None
    course_matrix = None
    course_embs = None

    if os.path.exists(vec_path) and os.path.exists(mat_path):
        with open(vec_path, 'rb') as f:
            course_vec = pickle.load(f)
        course_matrix = load_npz(mat_path)
        if course_matrix.shape[0] != len(courses_df):
            course_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=15000)
            course_matrix = course_vec.fit_transform(courses_df['course_text'])
    else:
        course_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=15000)
        course_matrix = course_vec.fit_transform(courses_df['course_text'])

    if os.path.exists(emb_path):
        course_embs = np.load(emb_path)
        if len(course_embs) != len(courses_df):
            course_embs = None

    return course_vec, course_matrix, course_embs

@st.cache_data(show_spinner=False)
def load_career_data():
    roadmap_path = os.path.join(DATA_DIR, 'career_roadmap.json')
    summary_path = os.path.join(DATA_DIR, 'career_track_summary.csv')
    tracks_path  = os.path.join(DATA_DIR, 'dynamic_career_tracks.json')
    roadmap, summary_df, tracks = None, None, None
    if os.path.exists(roadmap_path):
        with open(roadmap_path) as f: roadmap = json.load(f)
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
    if os.path.exists(tracks_path):
        with open(tracks_path) as f: tracks = json.load(f)
    return roadmap, summary_df, tracks

# ── Core logic ────────────────────────────────────────────────
def normalise_skill_query(skill):
    s = str(skill).strip().lower()
    mapping = {
        '.net': 'dotnet c# asp.net .net framework',
        'c#': 'c sharp c# programming',
        'c++': 'c++ programming',
        'ci/cd': 'ci cd devops pipeline',
        'power bi': 'power bi business intelligence dashboard',
        'data science': 'data science machine learning analytics',
        'data analysis': 'data analysis analytics excel sql',
    }
    return mapping.get(s, s)

def get_resume_profile_text(skills, education, roles, years_exp):
    parts = []
    if skills: parts.append('skills: ' + ', '.join(skills))
    if education: parts.append('education: ' + ' '.join(education))
    if roles: parts.append('experience: ' + ' '.join(roles))
    parts.append(f'years experience: {years_exp}')
    return ' '.join(parts)

def run_matching(resume_text, jobs_df, cols, sbert_model=None, tfidf_vec=None, job_tfidf=None, job_embs=None):
    if 'extract_resume_profile' in globals() and extract_resume_profile is not None:
        profile = extract_resume_profile(resume_text)
        skills = profile.get('skills', [])
        education = profile.get('education', [])
        roles = profile.get('roles', [])
        years_exp = profile.get('years_exp', 0.0)
    else:
        skills    = extract_skills(resume_text)
        education = extract_education(resume_text)
        roles     = extract_experience_roles(resume_text)
        years_exp = extract_years_experience(resume_text)
    skills_text = ', '.join(skills)
    skills_set  = set(s.lower() for s in skills)
    profile_text = get_resume_profile_text(skills, education, roles, years_exp)

    if tfidf_vec is None or job_embs is None:
        raise RuntimeError('Job indexes are missing. Run Objective 3 once to save the JobStreet indexes.')

    job_texts = jobs_df[cols['desc']].fillna('').astype(str).tolist()
    if job_tfidf is None:
        job_tfidf = tfidf_vec.transform(job_texts)

    ALPHA = 0.3
    resume_tfidf = tfidf_vec.transform([profile_text])
    tfidf_scores = cosine_similarity(resume_tfidf, job_tfidf).flatten()

    if sbert_model is None:
        sbert_model = load_sbert()
    resume_emb = sbert_model.encode([profile_text], normalize_embeddings=True)
    sbert_scores = (job_embs @ resume_emb.T).flatten()

    hybrid_scores = (1 - ALPHA) * tfidf_scores + ALPHA * sbert_scores

    # small boost for jobs that mention detected skills
    skill_boost = np.zeros(len(jobs_df))
    for sk in skills_set:
        if len(sk) < 2:
            continue
        try:
            contains = jobs_df[cols['desc']].fillna('').astype(str).str.lower().str.contains(r'\b' + re.escape(sk) + r'\b', regex=True)
            skill_boost += contains.astype(float).values * 0.01
        except Exception:
            pass
    hybrid_scores = hybrid_scores + np.minimum(skill_boost, 0.08)

    top_idx = np.argsort(-hybrid_scores)[:JOB_MATCH_LIMIT]
    top_jobs = jobs_df.iloc[top_idx].copy().reset_index(drop=True)
    top_jobs['tfidf_score']  = tfidf_scores[top_idx]
    top_jobs['sbert_score']  = sbert_scores[top_idx]
    top_jobs['hybrid_score'] = hybrid_scores[top_idx]

    gap_results = []
    for i, row in top_jobs.iterrows():
        desc = str(row.get(cols['desc'], ''))
        desc_lower = desc.lower()
        required = set()
        for sk in SKILL_KEYWORDS:
            if re.search(r'\b'+re.escape(sk)+r'\b', desc_lower):
                required.add(sk)
        matched = set()
        missing = set()
        for req in required:
            hit = skill_match(req, skills_set)
            (matched if hit else missing).add(req)
        match_pct = len(matched)/len(required)*100 if required else 0
        gap_results.append({
            'rank': i+1,
            'job_title': row.get(cols['title'], f'Job {i+1}'),
            'category': row.get(cols['cat'], 'N/A') if cols['cat'] else 'N/A',
            'company':  row.get(cols['company'], 'N/A') if cols['company'] else 'N/A',
            'hybrid_score': round(float(row['hybrid_score']),4),
            'tfidf_score':  round(float(row['tfidf_score']),4),
            'sbert_score':  round(float(row['sbert_score']),4),
            'match_pct':    round(match_pct,1),
            'gap_score':    round(1-match_pct/100,3),
            'matched_skills': clean_gap_skills(matched),
            'missing_skills': clean_gap_skills(missing),
            'required_skills': clean_gap_skills(required),
            'skill_coverage': f"{len(matched)}/{len(required)} skills" if required else '0/0 skills',
        })
    gap_df = pd.DataFrame(gap_results)
    return skills, education, roles, years_exp, gap_df

def get_course_recs(skill, course_vec, course_matrix, courses_df, course_cols, top_k=3, sbert_model=None, course_embs=None):
    query = normalise_skill_query(skill)
    sv = course_vec.transform([query])
    tfidf_sims = cosine_similarity(sv, course_matrix).flatten()

    if course_embs is not None and sbert_model is not None:
        skill_emb = sbert_model.encode([query], normalize_embeddings=True)
        sbert_sims = (course_embs @ skill_emb.T).flatten()
    else:
        sbert_sims = np.zeros_like(tfidf_sims)

    rating = courses_df['rating_norm'].values if 'rating_norm' in courses_df.columns else np.full(len(courses_df), 0.5)
    scores = 0.50 * tfidf_sims + 0.30 * sbert_sims + 0.20 * rating

    # keep results related to the skill text
    valid = np.where((tfidf_sims >= 0.02) | (sbert_sims >= 0.25))[0]
    if len(valid) == 0:
        return []

    tech_exact = {'.net','php','java','python','sql','mysql','react','angular','aws','azure','power bi','tableau','docker','kubernetes'}
    low_skill = str(skill).lower().strip()
    if low_skill in tech_exact:
        text = courses_df['course_text'].fillna('').astype(str).str.lower()
        if low_skill == '.net':
            mask = text.str.contains(r'\.net|dotnet|asp\.net|c#|c sharp', regex=True)
        else:
            mask = text.str.contains(r'\b' + re.escape(low_skill) + r'\b', regex=True)
        exact_idx = np.where(mask.values)[0]
        if len(exact_idx) > 0:
            valid = np.intersect1d(valid, exact_idx)
            if len(valid) == 0:
                valid = exact_idx

    ordered_idx = valid[np.argsort(-scores[valid])]
    valid_link_results = []
    backup_results = []

    for idx in ordered_idx[:max(top_k * 8, 20)]:
        row = courses_df.iloc[idx]
        course_name = str(row.get(course_cols['name'], 'N/A'))
        if not is_usable_course_title(course_name):
            continue
        raw_url = str(row.get(course_cols['url'], '')) if course_cols['url'] else ''
        fixed_url = valid_url(raw_url)
        item = {
            'course_name': course_name,
            'org':         str(row.get(course_cols['org'], 'N/A')) if course_cols['org'] else 'N/A',
            'diff':        str(row.get(course_cols['diff'], 'N/A')) if course_cols['diff'] else 'N/A',
            'rating':      str(row.get('rating_display', 'Not rated')),
            'url':         fixed_url,
            'sim':         round(float(scores[idx]), 3),
        }
        if fixed_url:
            valid_link_results.append(item)
        else:
            backup_results.append(item)
        if len(valid_link_results) >= top_k:
            break

    if valid_link_results:
        return valid_link_results[:top_k]
    return backup_results[:1]

def build_career_tracks(jobs_df, cols):
    SENIORITY_MAP = {
        'Junior':       ['junior','entry','graduate','trainee','intern','fresh','associate','assistant'],
        'Mid-Level':    ['mid','intermediate','experienced','specialist','executive'],
        'Senior':       ['senior','sr.','lead','principal','expert','advanced'],
        'Manager/Lead': ['manager','head','director','chief','vp','vice president']
    }
    def get_tier(title):
        t = title.lower()
        for tier in reversed(TIER_ORDER):
            if any(kw in t for kw in SENIORITY_MAP[tier]):
                return tier
        return 'Mid-Level'

    jobs_df = jobs_df.copy()
    jobs_df['tier'] = jobs_df[cols['title']].apply(get_tier)

    cat_col = cols['cat']
    if cat_col and cat_col in jobs_df.columns:
        cat_counts = jobs_df[cat_col].value_counts()
        valid_cats = cat_counts[cat_counts >= 20].index.tolist()[:10]
    else:
        valid_cats = ['All Roles']

    tracks = {}
    for cat in valid_cats:
        cat_jobs = jobs_df if cat == 'All Roles' else jobs_df[jobs_df[cat_col] == cat]
        tier_skills = {}
        for tier in TIER_ORDER:
            tier_jobs = cat_jobs[cat_jobs['tier'] == tier]
            if len(tier_jobs) >= 3:
                combined = ' '.join(tier_jobs[cols['desc']].fillna('').astype(str).str.lower().tolist())
                found = {}
                for sk in SKILL_KEYWORDS:
                    n = len(re.findall(r'\b'+re.escape(sk)+r'\b', combined))
                    if n >= 3: found[sk] = n
                tier_skills[tier] = [sk for sk,_ in sorted(found.items(),key=lambda x:-x[1])[:25]]
            else:
                tier_skills[tier] = []
        tracks[cat] = tier_skills
    return tracks

def score_career(user_set, tracks, years_exp):
    results = {}
    for cat, tier_skills in tracks.items():
        cumulative = set()
        tier_results = {}
        for tier in TIER_ORDER:
            cumulative |= set(tier_skills.get(tier,[]))
            if not cumulative:
                tier_results[tier] = {'match_pct':0,'matched':[],'missing':[]}
                continue
            matched = set()
            missing = set()
            for req in cumulative:
                hit = skill_match(req, user_set)
                (matched if hit else missing).add(req)
            match_pct = len(matched)/len(cumulative)*100
            tier_results[tier] = {
                'match_pct': round(match_pct,1),
                'matched':   clean_gap_skills(matched),
                'missing':   clean_gap_skills(missing)
            }
        current = 'Junior'
        for tier in TIER_ORDER:
            if tier_results.get(tier,{}).get('match_pct',0) >= 40:
                current = tier
        idx = TIER_ORDER.index(current)
        if years_exp >= 8 and idx < 2:   current = TIER_ORDER[2]
        elif years_exp >= 3 and idx < 1: current = TIER_ORDER[1]
        results[cat] = {'tiers': tier_results, 'current_tier': current}
    return results



def is_usable_course_title(name):
    name = str(name).strip()
    if not name or name.lower() in {'nan', 'none', 'n/a'}:
        return False
    letters = [c for c in name if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if ord(c) < 128]
    return len(ascii_letters) / max(len(letters), 1) >= 0.65

def track_domain_bonus(track, skills):
    track_low = str(track).lower()
    user = {str(s).lower().strip() for s in skills}
    bonus = 0
    for domain, kws in DOMAIN_SIGNAL_KEYWORDS.items():
        overlap = sum(1 for kw in kws if any(kw == u or kw in u or u in kw for u in user))
        if overlap == 0:
            continue
        if domain == 'marketing' and any(x in track_low for x in ['marketing','advertising','media','arts']):
            bonus += min(35, overlap * 6)
        elif domain == 'accounting' and any(x in track_low for x in ['accounting','finance','banking']):
            bonus += min(35, overlap * 6)
        elif domain in {'ict', 'tech'} and any(x in track_low for x in ['information','communication','technology','ict','computer','software','science']):
            bonus += min(45, overlap * 7)
        elif domain == 'customer_service' and any(x in track_low for x in ['call centre','customer','sales','retail']):
            bonus += min(20, overlap * 4)
    return bonus

def choose_career_path(career_df, career_scores, gap_df, user_skills=None):
    if career_df.empty:
        return None, 0, 'Junior', {}, None, career_df
    user_skills = user_skills or []

    # Use the job matching output as the main signal.
    # Stronger matched categories get a higher score than weak categories.
    category_scores = Counter()
    cat_candidates = ['category', 'Category', 'job_category', 'Job Category', 'classification', 'job_classification']
    cat_col = next((c for c in cat_candidates if c in gap_df.columns), None)
    if cat_col:
        for _, row in gap_df.head(10).iterrows():
            cat = str(row.get(cat_col, '')).strip()
            if not cat or cat.lower() == 'nan':
                continue
            category_scores[cat] += float(row.get('match_pct', row.get('Hybrid Match', row.get('hybrid_match', 0))))

    def category_bonus(track):
        track_low = str(track).lower()
        bonus = 0
        for cat, score in category_scores.items():
            cat_low = str(cat).lower()
            if track_low == cat_low or track_low in cat_low or cat_low in track_low:
                bonus += min(45, score / 4)
        return bonus

    ranked = career_df.copy()
    ranked['Category Bonus'] = ranked['Career Track'].apply(category_bonus)
    ranked['Resume Domain Bonus'] = ranked['Career Track'].apply(lambda t: track_domain_bonus(t, user_skills))
    ranked['Tier Skill Score'] = ranked[['Junior %','Mid-Level %','Senior %']].max(axis=1)
    # Final track should not be selected only by the tier percentage.
    # Job matching category and resume domain are given higher priority, so a CS resume stays in ICT.
    ranked['Selection Score'] = ranked['Tier Skill Score'] + (ranked['Category Bonus'] * 1.5) + ranked['Resume Domain Bonus']
    ranked = ranked.sort_values(['Selection Score', 'Category Bonus', 'Resume Domain Bonus', 'Tier Skill Score'], ascending=False).reset_index(drop=True)

    # Safety rule: if the resume clearly contains ICT/data skills and the matched jobs are mainly ICT,
    # prefer the ICT track over generic categories such as Advertising/Arts.
    tech_skills = {'python','java','c++','r','html','css','sql','mysql','ibm db2','data analysis','data science','machine learning'}
    user_low = {str(s).lower().strip() for s in user_skills}
    has_tech_profile = len(user_low & tech_skills) >= 4
    top_cat_text = ' '.join(str(c).lower() for c in category_scores.keys())
    if has_tech_profile and any(x in top_cat_text for x in ['information', 'communication', 'technology', 'ict', 'software', 'science']):
        ict_mask = ranked['Career Track'].astype(str).str.lower().str.contains('information|communication|technology|ict|computer|software|science', regex=True)
        if ict_mask.any():
            ict_rows = ranked[ict_mask].copy()
            non_ict_best = ranked.iloc[0]['Selection Score']
            # Put ICT first when it is reasonably supported by the resume and matched jobs.
            best_ict_idx = ict_rows['Selection Score'].idxmax()
            ranked.loc[best_ict_idx, 'Selection Score'] = max(float(ranked.loc[best_ict_idx, 'Selection Score']), float(non_ict_best) + 1)
            ranked = ranked.sort_values(['Selection Score', 'Category Bonus', 'Resume Domain Bonus', 'Tier Skill Score'], ascending=False).reset_index(drop=True)

    best_track = ranked.iloc[0]['Career Track']
    best_tier = ranked.iloc[0]['Current Tier']
    best_pct_col = f"{best_tier} %" if f"{best_tier} %" in ranked.columns else 'Tier Skill Score'
    best_pct = ranked.iloc[0].get(best_pct_col, ranked.iloc[0].get('Tier Skill Score', 0))
    best_tiers = career_scores[best_track]['tiers']
    tier_idx = TIER_ORDER.index(best_tier)
    next_tier = TIER_ORDER[tier_idx + 1] if tier_idx + 1 < len(TIER_ORDER) else None
    return best_track, best_pct, best_tier, best_tiers, next_tier, ranked

# ── UI helpers ────────────────────────────────────────────────
def badges(skills, cls='badge'):
    if not skills: return '<span style="color:#8b91b0;font-size:0.8rem">None</span>'
    return '<div class="badge-row">' + ''.join(
        f'<span class="badge {cls}">{s}</span>' for s in skills
    ) + '</div>'

def score_bar(pct, max_val=1.0):
    fill = min(pct/max_val, 1.0) * 100
    color = '#22c55e' if fill >= 60 else '#f59e0b' if fill >= 30 else '#ef4444'
    return f'''<div class="score-bar-bg">
      <div class="score-bar-fill" style="width:{fill:.1f}%;background:{color}"></div>
    </div>'''

def tier_badge(tier):
    cls = TIER_CLASS.get(tier, 'tier-junior')
    return f'<span class="tier-badge {cls}">{tier}</span>'



def make_result_pdf(analysis, courses_df=None, course_cols=None, course_vec=None, course_matrix=None, course_embs=None, sbert_model=None):
    # PDF summary for the uploaded resume result.
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    def clean(value):
        value = '' if value is None else str(value)
        value = re.sub(r'<[^>]+>', '', value)
        value = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return value

    def join_items(items, limit=None):
        items = list(items or [])
        if limit:
            items = items[:limit]
        return ', '.join(clean(x) for x in items) if items else 'None'

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name='Body', parent=styles['Normal'], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name='TitleBlue', parent=styles['Title'], fontSize=16, leading=20, textColor=colors.HexColor('#1f4e79')))
    styles.add(ParagraphStyle(name='H2Blue', parent=styles['Heading2'], fontSize=12, leading=15, textColor=colors.HexColor('#1f4e79'), spaceBefore=8, spaceAfter=5))

    story = []
    story.append(Paragraph('FYP Career Intelligence System', styles['TitleBlue']))
    story.append(Paragraph('Resume Analysis Output Summary', styles['Heading3']))
    story.append(Paragraph('This PDF summarizes the output generated from the uploaded resume in the dashboard.', styles['Body']))
    story.append(Spacer(1, 8))

    skills = analysis.get('skills', [])
    education = analysis.get('education', [])
    years_exp = analysis.get('years_exp', 0)
    gap_df = analysis.get('gap_df', pd.DataFrame())
    best_track = analysis.get('best_track', 'N/A')
    best_tier = analysis.get('best_tier', 'N/A')
    next_tier = analysis.get('next_tier') or 'Maintain / Advance'
    career_df = analysis.get('career_df', pd.DataFrame())

    best_match = float(gap_df['match_pct'].max()) if isinstance(gap_df, pd.DataFrame) and not gap_df.empty and 'match_pct' in gap_df.columns else 0
    avg_match = float(gap_df['match_pct'].mean()) if isinstance(gap_df, pd.DataFrame) and not gap_df.empty and 'match_pct' in gap_df.columns else 0

    summary_data = [
        ['Skills Detected', str(len(skills))],
        ['Best Job Match', f'{best_match:.1f}%'],
        ['Average Match', f'{avg_match:.1f}%'],
        ['Suggested Career Track', clean(best_track)],
        ['Current Tier', clean(best_tier)],
        ['Next Goal', clean(next_tier)],
        ['Experience Detected', f'{years_exp} years'],
    ]
    t = Table(summary_data, colWidths=[5.5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eef3f8')),
        ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#cccccc')),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)

    story.append(Paragraph('Extracted Resume Information', styles['H2Blue']))
    story.append(Paragraph('<b>Detected skills:</b> ' + join_items(skills), styles['Body']))
    story.append(Paragraph('<b>Education:</b> ' + join_items(education), styles['Body']))
    story.append(Spacer(1, 6))

    if isinstance(gap_df, pd.DataFrame) and not gap_df.empty:
        story.append(Paragraph(f'Top {min(len(gap_df), JOB_MATCH_LIMIT)} Matched Jobs', styles['H2Blue']))
        job_rows = [['Rank', 'Job Title', 'Company', 'Match', 'Skills']]
        for _, row in gap_df.head(JOB_MATCH_LIMIT).iterrows():
            job_rows.append([
                str(row.get('rank', '')),
                Paragraph(clean(row.get('job_title', 'N/A')), styles['Small']),
                Paragraph(clean(row.get('company', 'N/A')), styles['Small']),
                f"{float(row.get('match_pct', 0)):.1f}%",
                clean(row.get('skill_coverage', ''))
            ])
        jt = Table(job_rows, colWidths=[1.1*cm, 6.1*cm, 4.7*cm, 1.6*cm, 2.4*cm], repeatRows=1)
        jt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#cccccc')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7f9fb')]),
        ]))
        story.append(jt)
        story.append(Spacer(1, 6))
        story.append(Paragraph('Skill Gap Summary', styles['H2Blue']))
        for _, row in gap_df.head(5).iterrows():
            story.append(Paragraph(f"<b>Rank {row.get('rank')} - {clean(row.get('job_title', 'N/A'))}</b>", styles['Body']))
            story.append(Paragraph('You have: ' + join_items(row.get('matched_skills', []), 12), styles['Small']))
            story.append(Paragraph('You need: ' + join_items(row.get('missing_skills', []), 12), styles['Small']))
            story.append(Spacer(1, 3))

    if isinstance(gap_df, pd.DataFrame) and not gap_df.empty:
        missing_counts = Counter(sk for lst in gap_df['missing_skills'] for sk in lst)
        priority_missing = [skill for skill, _ in missing_counts.most_common(RECOMMENDATION_SKILL_LIMIT)]
        story.append(PageBreak())
        story.append(Paragraph('Learning Resource Recommendations', styles['H2Blue']))
        if courses_df is not None and course_vec is not None and course_matrix is not None and course_cols is not None:
            for skill in priority_missing:
                recs = get_course_recs(skill, course_vec, course_matrix, courses_df, course_cols, top_k=3, sbert_model=sbert_model, course_embs=course_embs)
                if not recs:
                    continue
                story.append(Paragraph(f'<b>{clean(skill.upper())}</b>', styles['Body']))
                for rec in recs:
                    link_status = 'Available' if valid_url(rec.get('url', '')) else 'Unavailable'
                    story.append(Paragraph(f"- {clean(rec.get('course_name', 'N/A'))} ({clean(rec.get('org', 'N/A'))}, {clean(rec.get('diff', 'N/A'))}, Rating: {clean(rec.get('rating', 'N/A'))}, Link: {link_status})", styles['Small']))
                story.append(Spacer(1, 3))
        else:
            story.append(Paragraph('Course dataset/index was not available when this PDF was generated.', styles['Body']))

    story.append(Paragraph('Career Path Recommendation', styles['H2Blue']))
    story.append(Paragraph(f'The suggested career track is <b>{clean(best_track)}</b>. The current tier is <b>{clean(best_tier)}</b> and the next goal is <b>{clean(next_tier)}</b>.', styles['Body']))
    best_tiers = analysis.get('best_tiers', {})
    focus_tier = next_tier if next_tier in best_tiers else best_tier
    if focus_tier in best_tiers:
        story.append(Paragraph(f'<b>Main focus for {clean(focus_tier)}</b>', styles['Body']))
        story.append(Paragraph('Supporting skills: ' + join_items(best_tiers[focus_tier].get('matched', []), 15), styles['Small']))
        story.append(Paragraph('Skills to improve: ' + join_items(best_tiers[focus_tier].get('missing', []), 15), styles['Small']))

    if isinstance(career_df, pd.DataFrame) and not career_df.empty:
        track_df = career_df.head(5).copy()
        tier_cols = [c for c in ['Junior %', 'Mid-Level %', 'Senior %'] if c in track_df.columns]
        if tier_cols:
            track_df['Skill Match %'] = track_df[tier_cols].apply(pd.to_numeric, errors='coerce').max(axis=1).round(0)
        cols = [c for c in ['Career Track', 'Current Tier', 'Skill Match %'] if c in track_df.columns]
        if cols:
            story.append(Spacer(1, 5))
            track_rows = [cols] + track_df[cols].astype(str).values.tolist()
            ct = Table(track_rows, colWidths=[7.5*cm, 4*cm, 3*cm])
            ct.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f4e79')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#cccccc')),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
            ]))
            story.append(ct)

    story.append(Spacer(1, 10))
    story.append(Paragraph('Note: This output is generated by an academic prototype. The recommendations are for guidance and do not replace professional career counselling or employer decisions.', styles['Small']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ── Shared extraction logic ───────────────────────────────────
# Extraction functions are kept in this file for the final Streamlit application.
extract_resume_profile = None
DOMAIN_SIGNAL_KEYWORDS = {
    'ict': {'python','java','c++','r','html','css','sql','mysql','ibm db2','data analysis','data science','machine learning','project management','javascript','power bi','tableau'},
    'accounting': {'accounting','financial reporting','bookkeeping','bank reconciliation','cash reconciliation','audit','auditing','payroll','statutory submissions','taxation','budgeting','forecasting','microsoft 365','million accounting system','auto count accounting system','quickbooks','accounts payable'},
    'marketing': {'digital marketing','seo','sem','social media marketing','social media management','content writing','google analytics','canva','campaign management','market research','brand awareness','customer engagement','crm','email marketing','copywriting'},
    'customer_service': {'customer service','sales','retail','cashier','storekeeper'}
}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem 0;">
      <div style="font-size:1.3rem;font-weight:700;color:#4f8ef7;letter-spacing:-0.5px;">🎯 FYP Career Intelligence System</div>
      <div style="font-size:0.75rem;color:#8b91b0;font-family:'DM Mono',monospace;">Research Prototype</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📌  Research Output Summary",
            "🏠  Prototype Demo",
            "📊  Job Matching",
            "📚  Recommendations",
            "🗺️  Career Path"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2d3148;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#8b91b0;'>Upload your resume to get started.</div>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Resume (PDF)", type=["pdf"], label_visibility="collapsed", key="resume_uploader")

    if uploaded is not None:
        import tempfile
        resume_bytes = uploaded.getvalue()
        resume_key = hashlib.md5(resume_bytes).hexdigest()
        if st.session_state.get('resume_key') != resume_key:
            old_path = st.session_state.get('resume_path')
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except Exception:
                    pass
            st.session_state.pop('analysis', None)
            st.session_state['resume_key'] = resume_key
            st.session_state['resume_name'] = uploaded.name
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(resume_bytes)
                st.session_state['resume_path'] = tmp.name
        st.markdown("<div class='status-ok'>✓ Resume uploaded</div>", unsafe_allow_html=True)
    else:
        if st.session_state.get('resume_key'):
            old_path = st.session_state.get('resume_path')
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except Exception:
                    pass
            for key in ['resume_key', 'resume_name', 'resume_path', 'analysis']:
                st.session_state.pop(key, None)

    st.markdown("<hr style='border-color:#2d3148;margin:1rem 0'>", unsafe_allow_html=True)

    course_status_paths = [
        os.path.join(DATA_DIR, 'coursera_cleaned.csv'),
    ]
    files_needed = {
        'JobStreet': os.path.join(DATA_DIR,'jobstreet_all_job_dataset.csv'),
        'Coursera': next((p for p in course_status_paths if os.path.exists(p)), course_status_paths[0]),
    }
    for label, path in files_needed.items():
        exists = os.path.exists(path)
        icon = "✓" if exists else "✗"
        cls  = "status-ok" if exists else "status-err"
        st.markdown(f"<div class='{cls}'>{icon} {label} dataset</div>", unsafe_allow_html=True)

    has_cache = os.path.exists(os.path.join(DATA_DIR,'job_embs.npy')) and os.path.exists(os.path.join(DATA_DIR,'job_tfidf_vec.pkl'))
    icon = "✓" if has_cache else "○"
    cls  = "status-ok" if has_cache else "status-err"
    st.markdown(f"<div class='{cls}'>{icon} Job indexes</div>", unsafe_allow_html=True)

    has_course_cache = os.path.exists(os.path.join(DATA_DIR,'course_tfidf_vec.pkl')) and os.path.exists(os.path.join(DATA_DIR,'course_tfidf_matrix.npz'))
    icon = "✓" if has_course_cache else "○"
    cls  = "status-ok" if has_course_cache else "status-err"
    st.markdown(f"<div class='{cls}'>{icon} Course indexes</div>", unsafe_allow_html=True)

# ── Navbar ────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div>
    <div class="navbar-brand">🎯 FYP Career Intelligence System</div>
    <div class="navbar-sub">AI-Driven Skill Gap Analysis & Career Path Recommendation System</div>
    <div class="navbar-sub" style="margin-top:0.25rem;font-size:0.68rem;color:#8b91b0;">The system name is uniquely generated for this academic project and does not correspond to any existing commercial system.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load shared data ──────────────────────────────────────────
jobs_df_full = None
jobs_cols    = None
if os.path.exists(os.path.join(DATA_DIR,'jobstreet_all_job_dataset.csv')):
    try:
        jobs_df_full, jobs_cols = load_jobs()
    except: pass

courses_df_full = None
course_cols     = None
course_vec      = None
course_matrix   = None
course_embs     = None
course_files_available = os.path.exists(os.path.join(DATA_DIR, 'coursera_cleaned.csv'))
if course_files_available:
    try:
        courses_df_full, course_cols = load_courses()
        course_vec, course_matrix, course_embs = load_course_index(courses_df_full)
    except Exception:
        pass

job_tfidf_vec = None
job_tfidf_matrix = None
job_embs = None
try:
    job_tfidf_vec, job_tfidf_matrix, job_embs = load_job_index()
except Exception:
    pass

sbert_model = None

# ── Run analysis if resume uploaded ──────────────────────────
if 'resume_path' in st.session_state and 'analysis' not in st.session_state:
    if jobs_df_full is not None:
        with st.spinner("Analysing resume..."):
            try:
                current_sbert = sbert_model
                if current_sbert is None:
                    with st.spinner("Loading matching model..."):
                        current_sbert = load_sbert()

                resume_text = read_pdf(st.session_state['resume_path'])
                skills, education, roles, years_exp, gap_df = run_matching(
                    resume_text, jobs_df_full, jobs_cols, current_sbert,
                    job_tfidf_vec, job_tfidf_matrix, job_embs)
                skills_set = set(s.lower() for s in skills)
                roadmap, summary_df, saved_tracks = load_career_data()
                career_tracks = saved_tracks if saved_tracks else build_career_tracks(jobs_df_full, jobs_cols)
                career_scores = score_career(skills_set, career_tracks, years_exp)
                summary_rows = []
                for cat, data in career_scores.items():
                    summary_rows.append({
                        'Career Track': cat,
                        'Current Tier': data['current_tier'],
                        'Junior %':    data['tiers'].get('Junior',{}).get('match_pct',0),
                        'Mid-Level %': data['tiers'].get('Mid-Level',{}).get('match_pct',0),
                        'Senior %':    data['tiers'].get('Senior',{}).get('match_pct',0),
                    })
                career_df = pd.DataFrame(summary_rows).sort_values('Junior %',ascending=False).reset_index(drop=True)
                best_track, best_pct, best_tier, best_tiers, next_tier, career_df = choose_career_path(
                    career_df, career_scores, gap_df, skills
                )
                st.session_state['analysis'] = {
                    'skills': skills, 'education': education, 'roles': roles,
                    'years_exp': years_exp, 'gap_df': gap_df,
                    'career_df': career_df, 'best_track': best_track,
                    'best_pct': best_pct, 'best_tier': best_tier,
                    'best_tiers': best_tiers, 'next_tier': next_tier,
                    'career_scores': career_scores, 'career_tracks': career_tracks
                }
            except Exception as e:
                st.error(f"Analysis error: {e}")

a = st.session_state.get('analysis', None)

# Downloadable PDF summary.
if a is not None:
    with st.sidebar:
        st.markdown("<hr style='border-color:#2d3148;margin:1rem 0'>", unsafe_allow_html=True)
        try:
            pdf_bytes = make_result_pdf(
                a,
                courses_df=courses_df_full,
                course_cols=course_cols,
                course_vec=course_vec,
                course_matrix=course_matrix,
                course_embs=course_embs,
                sbert_model=load_sbert() if course_embs is not None else None
            )
            safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', st.session_state.get('resume_name', 'resume_result')).strip('_')
            st.download_button(
                "Download PDF Result",
                data=pdf_bytes,
                file_name=f"{safe_name}_analysis_result.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception:
            st.caption("PDF download is not available for this run.")




def plot_internal_model_comparison_fallback():
    # Saved result from Objective 1 notebook using JobDataset and synthetic resumes.
    # This is shown when the CSV columns cannot be detected automatically.
    obj1_df = pd.DataFrame({
        "Model": ["TF-IDF", "BERT (DistilBERT)", "SBERT", "HYBRID"],
        "Top-100 Query Hit Rate (%)": [100.0, 62.1, 89.7, 100.0]
    })
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')
    colors = ["#4f8ef7" if m in ["TF-IDF", "HYBRID"] else "#2d3148" for m in obj1_df["Model"]]
    bars = ax.bar(obj1_df["Model"], obj1_df["Top-100 Query Hit Rate (%)"], color=colors)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", color="#e8eaf6", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_title("Objective 1 Model Comparison using JobDataset + Synthetic Resumes", color="#e8eaf6", fontsize=11)
    ax.set_ylabel("Top-100 Query Hit Rate (%)", color="#8b91b0")
    ax.tick_params(colors="#8b91b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2d3148')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("This graph is the saved Objective 1 result. TF-IDF and Hybrid achieved the highest top-100 query hit rate in the internal comparison.")

# ════════════════════════════════════════════════════════════
# PAGE: RESEARCH OUTPUT SUMMARY
# Research output summary page.
# ════════════════════════════════════════════════════════════
if page == "📌  Research Output Summary":
    st.markdown('<div class="section-header">Research Output Summary</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-color:#4f8ef7;background:#0d1829">
      <div class="card-title">Final Dashboard Purpose</div>
      <div style="font-size:0.92rem;color:#e8eaf6;line-height:1.65">
        This dashboard is the final interactive prototype for the project. It allows resume upload and shows the
        complete flow: resume extraction, hybrid job matching, skill gap analysis, course recommendation,
        and career path suggestion.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Research Workflow")
    workflow_items = ["Resume Upload", "Skill Extraction", "Hybrid Job Matching", "Skill Gap Analysis", "Course Recommendation", "Career Path Suggestion"]
    workflow_html = '<div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-bottom:1rem">'
    for i, item in enumerate(workflow_items, 1):
        workflow_html += f"""
        <div style="background:#1a1d27;border:1px solid #2d3148;border-radius:10px;padding:0.7rem 0.9rem;min-width:150px">
          <div style="font-size:0.68rem;color:#8b91b0;font-family:DM Mono,monospace">STEP {i}</div>
          <div style="font-size:0.88rem;color:#e8eaf6;font-weight:600">{item}</div>
        </div>"""
    workflow_html += '</div>'
    st.markdown(workflow_html, unsafe_allow_html=True)

    st.markdown("### Model Selection Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card"><div class="card-title">Selected Model</div>
        <div class="card-value" style="font-size:1.2rem">Hybrid</div>
        <div class="card-sub">TF-IDF + SBERT</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><div class="card-title">Formula</div>
        <div class="card-value" style="font-size:1rem;line-height:1.3">0.7 TF-IDF<br>+ 0.3 SBERT</div>
        <div class="card-sub">Higher weight is given to keyword overlap</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><div class="card-title">Reason</div>
        <div class="card-value" style="font-size:1.05rem;line-height:1.25">Keyword + Meaning</div>
        <div class="card-sub">Combines exact skills and semantic similarity</div></div>""", unsafe_allow_html=True)

    st.markdown("### Objective 1 Model Comparison")
    model_result_paths = [
        "merged_evaluation_results.csv", os.path.join(DATA_DIR, "merged_evaluation_results.csv"),
        "model_comparison_results.csv", os.path.join(DATA_DIR, "model_comparison_results.csv"),
        "objective1_model_comparison.csv", os.path.join(DATA_DIR, "objective1_model_comparison.csv"),
    ]
    model_result_path = next((p for p in model_result_paths if os.path.exists(p)), None)
    if model_result_path:
        try:
            model_df = pd.read_csv(model_result_path)
            lower_cols = {c.lower().strip(): c for c in model_df.columns}
            model_col = next((lower_cols[k] for k in ["model", "models", "method", "algorithm"] if k in lower_cols), None)
            metric_col = next((lower_cols[k] for k in ["accuracy", "f1", "f1-score", "f1_score", "precision", "score", "match_score"] if k in lower_cols), None)
            if model_col and metric_col:
                plot_df = model_df[[model_col, metric_col]].dropna().copy()
                plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
                plot_df = plot_df.dropna()
                if not plot_df.empty:
                    fig, ax = plt.subplots(figsize=(7, 3.2))
                    fig.patch.set_facecolor('#1a1d27')
                    ax.set_facecolor('#1a1d27')
                    labels = plot_df[model_col].astype(str)
                    vals = plot_df[metric_col]
                    bars = ax.bar(labels, vals, color="#4f8ef7")
                    for bar in bars:
                        h = bar.get_height()
                        label = f"{h:.2f}" if h <= 1 else f"{h:.1f}"
                        ax.text(bar.get_x() + bar.get_width()/2, h + (vals.max()*0.02), label, ha="center", color="#e8eaf6", fontsize=9, fontweight="bold")
                    ax.set_title("Internal Model Comparison using JobDataset + Synthetic Resumes", color="#e8eaf6", fontsize=11)
                    ax.set_ylabel(metric_col, color="#8b91b0")
                    ax.tick_params(colors="#8b91b0", labelsize=8)
                    for spine in ax.spines.values(): spine.set_color('#2d3148')
                    plt.xticks(rotation=15, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    plot_internal_model_comparison_fallback()
            else:
                plot_internal_model_comparison_fallback()
        except Exception:
            plot_internal_model_comparison_fallback()
    else:
        plot_internal_model_comparison_fallback()

    st.markdown("### RJDB Dataset External Validation")
    st.markdown("""
    <div class="card">
      <div style="font-size:0.9rem;color:#e8eaf6;line-height:1.65">
        RJDB is a resume-job description benchmark dataset from the literature. It contains job descriptions with
        matched and unmatched resumes. In this project, it is used as an external validation dataset for Objective 1
        to check whether the matching model gives a higher score to the matched resume.
      </div>
    </div>
    """, unsafe_allow_html=True)
    rjdb_summary_paths = ["rjdb_external_validation_summary.csv", os.path.join(DATA_DIR, "rjdb_external_validation_summary.csv")]
    rjdb_summary_path = next((p for p in rjdb_summary_paths if os.path.exists(p)), None)
    if rjdb_summary_path:
        rjdb_df = pd.read_csv(rjdb_summary_path)
        show_df = rjdb_df.copy()
        for col in ["Avg Matched Score", "Avg Unmatched Score", "Avg Score Difference", "Pairwise Accuracy (%)", "Runtime (sec)"]:
            if col in show_df.columns:
                show_df[col] = pd.to_numeric(show_df[col], errors="coerce").round(3)
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        if "Model" in rjdb_df.columns and "Pairwise Accuracy (%)" in rjdb_df.columns:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.patch.set_facecolor('#1a1d27')
            ax.set_facecolor('#1a1d27')
            plot_df = rjdb_df.copy()
            plot_df["Pairwise Accuracy (%)"] = pd.to_numeric(plot_df["Pairwise Accuracy (%)"], errors="coerce")
            labels = plot_df["Model"].astype(str).str.replace("Hybrid (0.7 TF-IDF + 0.3 SBERT)", "Hybrid", regex=False)
            bars = ax.bar(labels, plot_df["Pairwise Accuracy (%)"], color="#4f8ef7")
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", color="#e8eaf6", fontsize=9, fontweight="bold")
            ax.set_ylim(0, 100)
            ax.set_title("RJDB Dataset External Validation", color="#e8eaf6", fontsize=11)
            ax.set_ylabel("Pairwise Accuracy (%)", color="#8b91b0")
            ax.tick_params(colors="#8b91b0", labelsize=8)
            for spine in ax.spines.values(): spine.set_color('#2d3148')
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Place `rjdb_external_validation_summary.csv` in the app folder or data folder to show the RJDB validation result.")

    st.markdown("### Objective 2 Extraction Technique Selection")
    st.markdown("""
    <div class="card">
      <div style="font-size:0.9rem;color:#e8eaf6;line-height:1.65">
        The resume extraction stage compares three techniques: Rule-based / Keyword-based extraction,
        TF-IDF Keyword Extraction, and SBERT Similarity Extraction. The extracted skills are evaluated
        against a manual reference list prepared from the uploaded resume. Rule-based extraction is selected
        because it provides the highest validation score with controlled extraction output.
      </div>
    </div>
    """, unsafe_allow_html=True)

    extraction_df = pd.DataFrame({
        "Method": ["Rule-based / Keyword-based", "TF-IDF Keyword Extraction", "SBERT Similarity Extraction"],
        "Extracted Count": [18, 8, 32],
        "Correct Count": [18, 7, 16],
        "Missed Count": [0, 11, 2],
        "Noisy Count": [0, 1, 16],
        "Precision": [1.000, 0.875, 0.500],
        "Recall": [1.000, 0.389, 0.889],
        "F1-score": [1.000, 0.538, 0.640],
        "Final Decision": ["Selected", "Not selected", "Not selected"]
    })
    st.dataframe(extraction_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')
    labels = extraction_df["Method"].str.replace(" / ", "\n/ ", regex=False).str.replace(" Extraction", "", regex=False)
    bars = ax.bar(labels, extraction_df["F1-score"], color=["#4f8ef7", "#2d3148", "#2d3148"])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"{h:.3f}", ha="center", color="#e8eaf6", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("Objective 2 Extraction Technique Comparison", color="#e8eaf6", fontsize=11)
    ax.set_ylabel("F1-score", color="#8b91b0")
    ax.tick_params(colors="#8b91b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2d3148')
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Selected extraction method: Rule-based / Keyword-based extraction.")

    st.markdown("### Main Outputs by Objective")
    output_df = pd.DataFrame({
        "Objective": ["Obj 1", "Obj 1 Validation", "Obj 2", "Obj 3", "Obj 4", "Obj 5", "Obj 6"],
        "Input / Dataset": ["JobDataset + synthetic resumes", "RJDB", "Uploaded real resume", "JobStreet job dataset", "Coursera course dataset", "Skill gap and matching outputs", "Streamlit prototype"],
        "Main Output": ["Selected Hybrid model", "External validation result", "Selected rule-based extraction; extracted skills, education, experience", "Matched jobs, matched skills, missing skills", "Recommended courses for missing skills", "Career track, tier, next goal", "Interactive dashboard visualization"]
    })
    st.dataframe(output_df, use_container_width=True, hide_index=True)

    st.markdown("### Dataset Size Used in Prototype")
    d1, d2, d3 = st.columns(3)
    with d1:
        count = f"{len(jobs_df_full):,}" if jobs_df_full is not None else "Not loaded"
        st.markdown(f"""<div class="card"><div class="card-title">JobStreet Jobs</div><div class="card-value" style="font-size:1.25rem">{count}</div><div class="card-sub">Used for Objective 3</div></div>""", unsafe_allow_html=True)
    with d2:
        count = f"{len(courses_df_full):,}" if courses_df_full is not None else "Not loaded"
        st.markdown(f"""<div class="card"><div class="card-title">Course Records</div><div class="card-value" style="font-size:1.25rem">{count}</div><div class="card-sub">Used for Objective 4</div></div>""", unsafe_allow_html=True)
    with d3:
        count = "100 samples" if rjdb_summary_path else "CSV not loaded"
        st.markdown(f"""<div class="card"><div class="card-title">RJDB Validation</div><div class="card-value" style="font-size:1.25rem">{count}</div><div class="card-sub">Used for Objective 1 validation</div></div>""", unsafe_allow_html=True)

    st.markdown("### Final Research Link")
    st.markdown("""
    <div class="card" style="border-color:#4f8ef7;background:#0d1829">
      <div style="font-size:0.92rem;color:#e8eaf6;line-height:1.65">
        The dashboard presents the integrated research prototype. RJDB supports external validation for the matching model,
        while JobStreet and the Coursera course dataset support skill gap analysis and course recommendation outputs.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE: PROTOTYPE DEMO
# ════════════════════════════════════════════════════════════
elif page == "🏠  Prototype Demo":
    if a is None:
        st.markdown("""
        <div class="upload-zone">
          <h3>Upload your resume to get started</h3>
          <p>PDF resume input for skill gap and career path analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        # Prototype workflow steps.
        st.markdown('<div class="section-header">Prototype Demo Steps</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div class="card">
              <div class="card-title">Step 1</div>
              <div class="card-value" style="font-size:1rem">Upload Resume</div>
              <div class="card-sub">PDF resume input</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="card">
              <div class="card-title">Step 2</div>
              <div class="card-value" style="font-size:1rem">Run Analysis</div>
              <div class="card-sub">skills, jobs, gaps</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class="card">
              <div class="card-title">Step 3</div>
              <div class="card-value" style="font-size:1rem">View Outputs</div>
              <div class="card-sub">recommendations and career path</div>
            </div>""", unsafe_allow_html=True)
    else:
        # Summary cards
        gap_df = a['gap_df']
        avg_match = gap_df['match_pct'].mean()
        top_match = gap_df['match_pct'].max()
        all_missing = []
        for lst in gap_df['missing_skills']: all_missing.extend(lst)

        st.markdown('<div class="section-header">Profile Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="card">
              <div class="card-title">Skills Detected</div>
              <div class="card-value">{len(a['skills'])}</div>
              <div class="card-sub">From resume</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card">
              <div class="card-title">Best Match</div>
              <div class="card-value">{top_match:.0f}%</div>
              <div class="card-sub">Top job match</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="card">
              <div class="card-title">Career Track</div>
              <div class="card-value" style="font-size:1rem;line-height:1.3">{a['best_track']}</div>
              <div class="card-sub">{a['best_pct']:.0f}% match</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="card">
              <div class="card-title">Current Tier</div>
              <div class="card-value" style="font-size:1.2rem">{a['best_tier']}</div>
              <div class="card-sub">Next: {a['next_tier'] or 'Top level'}</div>
            </div>""", unsafe_allow_html=True)

        st.caption("Best Match shows the strongest job match from the uploaded resume. Career Track % shows the overall readiness for the suggested career direction, so both values can be different.")

        # Two-column layout
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown('<div class="section-header">Your Skills</div>', unsafe_allow_html=True)
            st.markdown(badges(a['skills'], 'badge badge-blue'), unsafe_allow_html=True)

            st.markdown('<div class="section-header">Education</div>', unsafe_allow_html=True)
            for edu in a['education']:
                st.markdown(f"<div style='font-size:0.85rem;color:#e8eaf6;padding:0.3rem 0'>{edu}</div>",
                            unsafe_allow_html=True)

            st.markdown('<div class="section-header">Experience</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.9rem;color:#e8eaf6;padding:0.2rem 0'>{a['years_exp']} years detected</div>", unsafe_allow_html=True)
            for role in a['roles']:
                st.markdown(f"<div style='font-size:0.85rem;color:#e8eaf6;padding:0.2rem 0'>💼 {role}</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-header">Top Missing Skills</div>', unsafe_allow_html=True)
            missing_counts = Counter(all_missing).most_common(10)
            if missing_counts:
                mc_df = pd.DataFrame(missing_counts, columns=['Skill','Count'])
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#1a1d27')
                ax.set_facecolor('#1a1d27')
                ax.barh(mc_df['Skill'], mc_df['Count'], color='#ef4444', alpha=0.85)
                ax.tick_params(colors='#8b91b0', labelsize=8)
                ax.spines['bottom'].set_color('#2d3148')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#2d3148')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ════════════════════════════════════════════════════════════
# PAGE: RESUME ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📊  Job Matching":
    st.markdown(f'<div class="section-header">Top {JOB_MATCH_LIMIT} Matched Jobs</div>', unsafe_allow_html=True)

    if a is None:
        st.info("Upload your resume from the sidebar to see job matches.")
    else:
        gap_df = a['gap_df']

        # Summary stats
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="card">
              <div class="card-title">Avg Match</div>
              <div class="card-value">{gap_df['match_pct'].mean():.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card">
              <div class="card-title">Best Match</div>
              <div class="card-value">{gap_df['match_pct'].max():.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="card">
              <div class="card-title">Avg Skill Gap</div>
              <div class="card-value">{gap_df['gap_score'].mean()*100:.1f}%</div>
              <div class="card-sub">Lower means fewer missing skills</div>
            </div>""", unsafe_allow_html=True)

        st.caption("Ranks are selected using the Hybrid model score. The match percentage shows skill coverage, so the percentages may not always appear in descending order.")

        # Job cards
        for _, row in gap_df.iterrows():
            pct   = row['match_pct']
            color = '#22c55e' if pct >= 60 else '#f59e0b' if pct >= 30 else '#ef4444'
            with st.expander(f"Rank {row['rank']} — {row['job_title']} ({pct:.0f}% match)", expanded=False):
                cc1, cc2 = st.columns([2,1])
                with cc1:
                    st.markdown(f"**Company:** {row['company']}")
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown("**Required Skills:**")
                    st.markdown(badges(row['required_skills'], 'badge'), unsafe_allow_html=True)
                    st.markdown("**You Have:**")
                    st.markdown(badges(row['matched_skills'], 'badge badge-green'), unsafe_allow_html=True)
                    st.markdown("**You Need:**")
                    st.markdown(badges(row['missing_skills'], 'badge badge-red'), unsafe_allow_html=True)
                with cc2:
                    st.markdown(f"""
                    <div style="text-align:center;padding:1rem">
                      <div style="font-size:2.5rem;font-weight:700;color:{color}">{pct:.0f}%</div>
                      <div style="color:#8b91b0;font-size:0.8rem">Hybrid Match</div>
                      <div style="margin-top:1rem;font-size:0.8rem;color:#8b91b0">
                        {row.get('skill_coverage', '')}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Match % chart
        st.markdown('<div class="section-header">Skill Match Overview</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,4))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        colors_bar = ['#22c55e' if p>=60 else '#f59e0b' if p>=30 else '#ef4444'
                      for p in gap_df['match_pct']]
        ax.bar([f"R{r}" for r in gap_df['rank']], gap_df['match_pct'], color=colors_bar)
        ax.axhline(60, color='#22c55e', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(30, color='#f59e0b', linestyle='--', alpha=0.5, linewidth=1)
        ax.tick_params(colors='#8b91b0')
        ax.set_ylabel('Match %', color='#8b91b0')
        for spine in ax.spines.values():
            spine.set_color('#2d3148')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ════════════════════════════════════════════════════════════
elif page == "📚  Recommendations":
    st.markdown('<div class="section-header">Learning Resource Recommendations</div>', unsafe_allow_html=True)

    if a is None:
        st.info("Upload your resume from the sidebar to see course recommendations.")
    elif courses_df_full is None or course_vec is None:
        st.warning("Coursera dataset not found. Place `coursera_cleaned.csv` in the `data/` folder.")
    else:
        gap_df = a['gap_df']
        all_missing = sorted(set(
            sk for lst in gap_df['missing_skills'] for sk in lst
        ))

        if not all_missing:
            st.success("No skill gaps detected — you match all job requirements!")
        else:
            missing_counts = Counter(sk for lst in gap_df['missing_skills'] for sk in lst)
            priority_missing = [skill for skill, _ in missing_counts.most_common(RECOMMENDATION_SKILL_LIMIT)]
            st.markdown(f"<div style='color:#8b91b0;font-size:0.85rem;margin-bottom:1rem'>"
                        f"Showing priority courses for {len(priority_missing)} important missing skills across the top {JOB_MATCH_LIMIT} job matches. The filter can be used to view one job result.</div>",
                        unsafe_allow_html=True)
            st.caption("Courses are matched using missing-skill keywords and course descriptions.")

            # Filter by job
            job_options = ['All Jobs'] + [f"Rank {r} — {t}" for r, t in
                           zip(gap_df['rank'], gap_df['job_title'])]
            selected_job = st.selectbox("Filter by job", job_options)

            if selected_job == 'All Jobs':
                show_missing = priority_missing
            else:
                rank = int(selected_job.split(' ')[1])
                row  = gap_df[gap_df['rank'] == rank].iloc[0]
                show_missing = row['missing_skills']

            for skill in show_missing:
                recs = get_course_recs(skill, course_vec, course_matrix,
                                       courses_df_full, course_cols, top_k=3,
                                       sbert_model=load_sbert(), course_embs=course_embs)
                if not recs:
                    continue
                st.markdown(f"""<div style="margin-top:1rem">
                  <div class="section-header" style="font-size:0.85rem">
                    📌 {skill.upper()}
                  </div>""", unsafe_allow_html=True)
                for r in recs:
                    safe_url = valid_url(r.get('url', ''))
                    url_part = f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" style="color:#4f8ef7;font-size:0.75rem">View Course →</a>' if safe_url else '<span style="color:#8b91b0;font-size:0.75rem">Link unavailable</span>'
                    st.markdown(f"""
                    <div class="course-card">
                      <div class="course-name">{r['course_name']}</div>
                      <div class="course-meta">
                        🏫 {r['org']} &nbsp;|&nbsp; 
                        📊 {r['diff']} &nbsp;|&nbsp; 
                        ⭐ {r['rating']} &nbsp;|&nbsp;
                        {url_part}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE: CAREER PATH
# Career path page.
# ════════════════════════════════════════════════════════════
elif page == "🗺️  Career Path":
    st.markdown('<div class="section-header">Career Path Recommendation</div>', unsafe_allow_html=True)

    if a is None:
        st.info("Upload your resume from the sidebar to see career path analysis.")
    else:
        career_df = a['career_df']
        best_track = a['best_track']
        best_tier  = a['best_tier']
        best_tiers = a['best_tiers']
        next_tier  = a['next_tier']

        st.markdown("""
        <div class="card" style="border-color:#4f8ef7;background:#0d1829">
          <div class="card-title">How to read this page</div>
          <div style="font-size:0.9rem;color:#e8eaf6;line-height:1.65">
            This page gives a suggested career direction from the resume skills and job matching results.
            The highlighted tier shows the current readiness level. The next goal shows the recommended level to improve towards.
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="card"><div class="card-title">Suggested Track</div><div class="card-value" style="font-size:1.05rem;line-height:1.35">{best_track}</div><div class="card-sub">Best aligned career area</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card"><div class="card-title">Current Tier</div><div class="card-value" style="font-size:1.2rem">{best_tier}</div><div class="card-sub">Highlighted in the chart</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="card"><div class="card-title">Next Goal</div><div class="card-value" style="font-size:1.2rem">{next_tier or 'Maintain / Advance'}</div><div class="card-sub">Recommended improvement target</div></div>""", unsafe_allow_html=True)

        st.markdown("### Why this track was suggested")
        st.markdown(f"""
        <div class="card"><div style="font-size:0.9rem;color:#e8eaf6;line-height:1.65">
            <b>{best_track}</b> was selected because it has the strongest alignment with the detected resume skills
            and the categories from the matched jobs. The tier is estimated using skill coverage and experience indicators.
        </div></div>
        """, unsafe_allow_html=True)

        st.markdown("### Career Tier Overview")
        tier_names = [t for t in TIER_ORDER if t in best_tiers]
        tier_pcts  = [best_tiers[t]['match_pct'] for t in tier_names]
        fig, ax = plt.subplots(figsize=(8,3))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        bar_colors = ['#4f8ef7' if t == best_tier else '#2d3148' for t in tier_names]
        bars = ax.bar(tier_names, tier_pcts, color=bar_colors, edgecolor='none')
        for i, (t, pct) in enumerate(zip(tier_names, tier_pcts)):
            label = f"{pct:.0f}%" + ("  CURRENT" if t == best_tier else "")
            ax.text(i, pct + 2, label, ha='center', fontsize=8.5, color='#e8eaf6', fontweight='bold')
        ax.axhline(40, color='#f59e0b', linestyle='--', alpha=0.7, linewidth=1.2)
        ax.text(len(tier_names)-0.5, 42, '40% readiness reference', color='#f59e0b', fontsize=8, ha='right')
        ax.text(0.02, 0.93, 'Blue bar = current tier', transform=ax.transAxes, color='#4f8ef7', fontsize=8, fontweight='bold')
        ax.text(0.02, 0.86, 'Dashed line = reference only, not pass/fail', transform=ax.transAxes, color='#8b91b0', fontsize=8)
        ax.tick_params(colors='#8b91b0')
        ax.set_ylabel('Skill Match %', color='#8b91b0')
        ax.set_ylim(0, 100)
        for spine in ax.spines.values(): spine.set_color('#2d3148')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("The current tier is estimated using both skill match and resume experience. The percentages show skill coverage only, so the highest bar is not always selected as the final tier.")

        focus_tier = next_tier if next_tier and next_tier in best_tiers else best_tier
        focus_missing = best_tiers.get(focus_tier, {}).get('missing', [])[:8]
        focus_matched = best_tiers.get(focus_tier, {}).get('matched', [])[:8]

        st.markdown(f"### Main Focus for {focus_tier}")
        col_have, col_need = st.columns(2)
        with col_have:
            st.markdown("**Skills already supporting this level**")
            st.markdown(badges(focus_matched, 'badge badge-green'), unsafe_allow_html=True)
        with col_need:
            st.markdown("**Skills to improve next**")
            st.markdown(badges(focus_missing, 'badge badge-red'), unsafe_allow_html=True)

        st.markdown("### Other Possible Tracks")
        if career_df is not None and not career_df.empty:
            track_df = career_df.head(5).copy()
            tier_cols = [c for c in ['Junior %', 'Mid-Level %', 'Senior %'] if c in track_df.columns]
            if tier_cols:
                track_df['Skill Match %'] = track_df[tier_cols].apply(pd.to_numeric, errors='coerce').max(axis=1).round(0)
            cols = [col for col in ['Career Track', 'Current Tier', 'Skill Match %'] if col in track_df.columns]
            if cols:
                st.caption("These are alternative career directions based on the same resume skills. Higher skill match means stronger readiness for that track.")
                st.dataframe(track_df[cols], use_container_width=True, hide_index=True)
