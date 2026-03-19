# DRONA — Student Intelligence Suite

> *"DRONA doesn't give you answers. It drives you to find them."*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://drona-student-intelligence-suite.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/Vectorstore-HuggingFace-yellow)](https://huggingface.co/datasets/spark-2026/drona-vectorstore)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Likhitha-26-AI/drona-student-intelligence-suite)
[![NCERT](https://img.shields.io/badge/Knowledge%20Base-75%20NCERT%20PDFs-green)](https://ncert.nic.in)

---

## What is DRONA?

DRONA is an AI-powered K-12 EdTech platform built to mirror the vision of **Eklavya Solution's DRONA ecosystem** — India's first intelligent hybrid school platform backed by Microsoft.

Unlike generic chatbots that simply answer questions, DRONA uses **Socratic questioning** — guiding students to discover answers themselves, just like the legendary teacher Dronacharya taught his students through challenge, not instruction.

---

## Problem Statement

- Over **250 million K-12 students** in India have no intelligent academic support after school hours
- Teachers have **zero visibility** into which concepts confuse students most
- School administrators cannot identify **struggling classes** until it's too late
- Generic AI chatbots give direct answers — destroying the learning process

---

## Solution

DRONA connects **students, teachers, and school administrators** through three intelligent modules:

| Role | Feature | What It Does |
|---|---|---|
| Student | Doubt Resolution Engine | RAG-powered Socratic AI tutor over NCERT content |
| Teacher | Class Analytics Dashboard | Real-time doubt patterns, at-risk student alerts |
| Admin | School Intelligence Hub | Subject difficulty heatmap, predictive alerts |

---

## Architecture

```
75 NCERT PDFs (Class 9 & 10 — Science, Maths, History)
        |
        v
   PyPDF Loader → RecursiveCharacterTextSplitter (500 tokens)
        |
        v
   HuggingFace Embeddings (all-MiniLM-L6-v2)
        |
        v
   FAISS Vectorstore — 5,162 knowledge chunks
        |
        v
   Student types doubt → Similarity Search (top 3 chunks)
        |
        v
   Qwen 2.5 7B (HuggingFace Inference API)
        |
   [Turn 1-2] Socratic guiding question
   [Turn 3]   Full explanation with NCERT reference
   [Turn 4]   Teacher escalation triggered
        |
        v
   Response logged to SQLite
        |
        v
   Teacher Dashboard ← Doubt analytics, at-risk flags
   Admin Dashboard   ← School-wide heatmap, predictions
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| RAG Framework | LangChain + FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Qwen/Qwen2.5-7B-Instruct (HuggingFace) |
| Knowledge Base | 75 NCERT PDFs — 5,162 chunks |
| Frontend | Streamlit |
| Analytics | Plotly + Pandas |
| Database | SQLite |
| Deployment | Streamlit Cloud |
| Vector Storage | HuggingFace Hub (Dataset) |

---

## Platform Statistics

| Metric | Value |
|---|---|
| NCERT PDFs Indexed | 75 |
| Knowledge Chunks | 5,162 |
| Subjects Covered | Science, Mathematics, History, General |
| Classes | Class 9 and Class 10 |
| Stakeholder Roles | Student, Teacher, Admin |
| Deployment | Live on Streamlit Cloud |

---

## Key Features

### Socratic AI Tutor
- Student asks any NCERT doubt in natural language
- DRONA retrieves the top 3 most relevant NCERT chunks using FAISS
- Instead of answering directly, DRONA asks a guiding question
- After 3 exchanges, provides full explanation with NCERT reference
- After 4 exchanges, automatically notifies the teacher

### Teacher Analytics Dashboard
- Real-time KPIs: total doubts, resolved by AI, escalated, resolution rate
- Subject-wise doubt distribution bar chart
- Resolution breakdown pie chart
- At-risk student table with subject and doubt details
- Predictive alert for highest-volume subject

### Admin School Intelligence Hub
- School-wide subject difficulty heatmap (class × subject)
- Doubt volume trend over time (area chart)
- Active students, escalation rate, subject count metrics
- Predictive alert for worst-performing class-subject combination
- Complete doubt log with all student interactions

---

## How to Run Locally

### Prerequisites
- Python 3.11+
- HuggingFace account with Write token
- NCERT PDFs downloaded from ncert.nic.in

### Setup

```bash
git clone https://github.com/Likhitha-26-AI/drona-student-intelligence-suite
cd drona-student-intelligence-suite
pip install -r requirements.txt
```

Create a `.env` file:
```
HF_TOKEN=hf_your_token_here
```

Run the app:
```bash
streamlit run app.py
```

---

## How to Deploy (Full Guide)

A complete step-by-step Colab notebook is included:
**`DRONA_Student_Intelligence_Suite.ipynb`**

It covers:
- Installing all dependencies
- Building the FAISS vectorstore from NCERT PDFs
- Uploading vectorstore to HuggingFace Hub
- Writing and testing the Streamlit app
- Pushing to GitHub
- Deploying on Streamlit Cloud

### Hiding Your Token Safely

**In Google Colab:**
1. Click the key icon in the left sidebar
2. Add secret: Name = `HF_TOKEN`, Value = your token
3. Toggle Notebook access ON

**In Streamlit Cloud:**
1. App Settings → Secrets
2. Add: `HF_TOKEN = "hf_your_token_here"`

**Never paste your token directly in code.**

---

## Test Doubts

Try these in the live app:

| Class | Subject | Doubt |
|---|---|---|
| Class 9 | Science | Why does photosynthesis require sunlight? |
| Class 10 | Science | What is the difference between exothermic and endothermic reactions? |
| Class 9 | Mathematics | How do I solve a quadratic equation using the quadratic formula? |
| Class 10 | History | What caused the rise of nationalism in Europe? |
| Class 9 | Science | What is osmosis and how is it different from diffusion? |

---

## Project Structure

```
drona-student-intelligence-suite/
│
├── app.py                              ← Complete Streamlit application
├── requirements.txt                    ← Python dependencies
├── .streamlit/
│   └── config.toml                     ← Streamlit theme configuration
├── DRONA_Student_Intelligence_Suite.ipynb  ← Full deployment guide
└── README.md                           ← This file
```

---

## Why This Project

DRONA directly mirrors what **Eklavya Solution** is building — India's first intelligent hybrid K-12 ecosystem backed by Microsoft. Every feature in this project was designed to solve a real problem their platform faces:

| Eklavya's Goal | This Project's Implementation |
|---|---|
| AI-powered student support | Socratic RAG doubt engine |
| Teacher visibility | Real-time analytics dashboard |
| School-wide intelligence | Admin heatmap with predictions |
| Multi-stakeholder design | 3 role-based views |
| Curriculum-aware AI | RAG over actual NCERT content |

---

## Author

**Gundavaram Likhitha Rao**
B.Tech CSE-AIML 
