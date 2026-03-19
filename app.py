import streamlit as st
import os
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient, snapshot_download

st.set_page_config(
    page_title="DRONA - Student Intelligence Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap");
html, body, [class*="css"] { font-family: "Inter", sans-serif; }
.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 3rem 2rem; border-radius: 16px;
    text-align: center; color: white; margin-bottom: 2rem;
}
.hero-title { font-size: 2.8rem; font-weight: 700; margin-bottom: 0.3rem; }
.hero-sub { font-size: 1rem; color: #a0aec0; margin-bottom: 1.5rem; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 0.3rem 0.8rem; border-radius: 20px;
    font-size: 0.78rem; margin: 0.2rem; color: white;
}
.feature-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;
}
.feature-title { font-size: 0.95rem; font-weight: 600; color: #1a202c; margin-bottom: 0.3rem; }
.feature-desc { font-size: 0.82rem; color: #718096; line-height: 1.5; }
.metric-box {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.2rem;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #1a202c; }
.metric-lbl { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }
.chat-user {
    background: #EBF8FF; border-left: 3px solid #3182ce;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; font-size: 0.9rem;
}
.chat-ai {
    background: #F0FFF4; border-left: 3px solid #38a169;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; font-size: 0.9rem;
}
.chat-esc {
    background: #FFF5F5; border-left: 3px solid #e53e3e;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; font-size: 0.9rem;
}
.alert-box {
    background: #FFF5F5; border: 1px solid #feb2b2;
    border-radius: 8px; padding: 0.8rem 1rem; color: #c53030; font-size: 0.85rem; margin: 0.5rem 0;
}
.success-box {
    background: #F0FFF4; border: 1px solid #9ae6b4;
    border-radius: 8px; padding: 0.8rem 1rem; color: #276749; font-size: 0.85rem; margin: 0.5rem 0;
}
.turn-bar {
    background: #ebf8ff; border-radius: 8px;
    padding: 0.5rem 1rem; font-size: 0.8rem; color: #2b6cb0; margin-top: 0.5rem;
}
div[data-testid="stSidebar"] { background: #1a202c; }
div[data-testid="stSidebar"] * { color: white !important; }
.stButton > button {
    background: #2b6cb0; color: white; border: none;
    border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 500; width: 100%;
}
</style>
""", unsafe_allow_html=True)


def init_db():
    conn = sqlite3.connect("drona.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doubt_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT, student_name TEXT, class_level TEXT,
            subject TEXT, doubt_text TEXT, ai_response TEXT,
            timestamp TEXT, resolved INTEGER DEFAULT 0,
            escalated INTEGER DEFAULT 0, turn_count INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()

init_db()


@st.cache_resource
def load_vectorstore():
    if not os.path.exists("drona_vectorstore"):
        snapshot_download(
            repo_id="spark-2026/drona-vectorstore",
            repo_type="dataset",
            local_dir="drona_vectorstore",
            token=os.environ.get("HF_TOKEN", "")
        )
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "drona_vectorstore", emb,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_client():
    return InferenceClient(
        token=os.environ.get("HF_TOKEN", "")
    )


def detect_subject(text):
    t = text.lower()
    if any(w in t for w in ["photosynthesis","cell","atom","force","acid",
                             "reaction","tissue","energy","newton","chemical",
                             "element","osmosis","enzyme","mitochondria"]):
        return "Science"
    elif any(w in t for w in ["equation","triangle","polynomial","theorem",
                               "angle","circle","probability","algebra",
                               "fraction","integer","ratio","geometry"]):
        return "Mathematics"
    elif any(w in t for w in ["revolution","war","independence","nationalism",
                               "colonial","emperor","treaty","movement",
                               "gandhi","british","french"]):
        return "History"
    return "General"


def get_ai_response(client, doubt, context, history, turn_count):
    if turn_count >= 3:
        system = (
            "You are DRONA, a friendly AI tutor for Indian K-12 students. "
            "Use this NCERT content to help: " + context[:1000] +
            " Give a clear complete explanation in simple language. Under 120 words."
        )
    else:
        system = (
            "You are DRONA, a Socratic AI tutor for Indian K-12 students. "
            "NCERT content: " + context[:800] +
            " Previous chat: " + history[-500:] +
            " Ask ONE guiding question. Do NOT give direct answer. "
            "Be encouraging and friendly. Under 80 words."
        )
    try:
        resp = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": doubt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"I am having trouble connecting right now. Please try again in a moment. Error: {str(e)[:100]}"


def log_doubt(sid, name, cls, subj, doubt, resp, turns, esc=0):
    try:
        conn = sqlite3.connect("drona.db")
        conn.execute(
            "INSERT INTO doubt_logs (student_id,student_name,class_level,"
            "subject,doubt_text,ai_response,timestamp,resolved,escalated,turn_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, name, cls, subj, doubt, resp,
             datetime.now().isoformat(),
             1 if turns >= 3 else 0, esc, turns)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_data():
    try:
        conn = sqlite3.connect("drona.db")
        df = pd.read_sql_query("SELECT * FROM doubt_logs", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def demo_data():
    import numpy as np
    np.random.seed(42)
    names = ["Arjun Sharma","Priya Patel","Ravi Kumar","Sneha Reddy",
             "Amit Singh","Kavya Nair","Rohan Gupta","Divya Menon"]
    doubts = [
        "Why does photosynthesis require sunlight?",
        "How do I solve quadratic equations?",
        "What caused the French Revolution?",
        "What is the difference between acids and bases?",
        "How do I find the area of a circle?",
        "What is osmosis?",
        "What is Newton third law?",
        "What is nationalism?"
    ]
    return pd.DataFrame({
        "student_id": [f"S{i:03d}" for i in range(1,51)],
        "student_name": np.random.choice(names, 50),
        "class_level": np.random.choice(["Class 9","Class 10"], 50),
        "subject": np.random.choice(["Science","Mathematics","History","General"], 50),
        "doubt_text": np.random.choice(doubts, 50),
        "resolved": np.random.choice([0,1], 50, p=[0.3,0.7]),
        "escalated": np.random.choice([0,1], 50, p=[0.8,0.2]),
        "turn_count": np.random.randint(1,5,50),
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="3H").astype(str)
    })


with st.sidebar:
    st.markdown("### DRONA")
    st.markdown("*Student Intelligence Suite*")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Home", "Student", "Teacher", "Admin"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown(
        "<span style='font-size:0.75rem;color:#a0aec0;'>"
        "75 NCERT PDFs indexed<br>"
        "5,162 knowledge chunks<br>"
        "Powered by Mistral 7B<br>"
        "Built for Eklavya Solution"
        "</span>",
        unsafe_allow_html=True
    )


if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">DRONA</div>
        <div style="font-size:1.1rem;font-weight:500;margin-bottom:0.5rem;">Student Intelligence Suite</div>
        <div class="hero-sub">AI-powered platform connecting students, teachers, and administrators<br>through intelligent doubt resolution and real-time learning analytics.</div>
        <span class="badge">RAG over NCERT</span>
        <span class="badge">Socratic AI Tutor</span>
        <span class="badge">Real-Time Analytics</span>
        <span class="badge">Mistral 7B</span>
        <span class="badge">Multi-Role Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Platform Capabilities")
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Student", "Doubt Resolution Engine",
         "Ask any NCERT doubt. DRONA guides you using Socratic questioning. Escalates to teacher when stuck."),
        ("Teacher", "Class Analytics Dashboard",
         "See which topics confuse students. Get alerted on at-risk students. Know where to focus."),
        ("Admin", "School Intelligence Hub",
         "School-wide difficulty heatmap. Predictive alerts for struggling classes. Full engagement metrics."),
        ("AI", "RAG Pipeline",
         "75 NCERT PDFs indexed into 5,162 chunks. FAISS vector search. Curriculum-aware responses only."),
    ]
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size:1.4rem;margin-bottom:0.4rem;color:#2b6cb0;">{icon[0]}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Platform Statistics")
    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        ("75","NCERT PDFs"),("5,162","Knowledge Chunks"),
        ("6","Subjects Covered"),("4","Stakeholder Roles")
    ]):
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Select a role from the sidebar to get started.")


elif page == "Student":
    st.markdown("#### Doubt Resolution Engine")
    st.caption("Ask any doubt from your NCERT syllabus. DRONA will guide you to the answer.")

    c1, c2, c3 = st.columns(3)
    with c1:
        student_name = st.text_input("Your Name", placeholder="Enter your name")
    with c2:
        student_id = st.text_input("Student ID", placeholder="e.g. S001")
    with c3:
        class_level = st.selectbox("Class", ["Class 9", "Class 10"])

    if not student_name or not student_id:
        st.markdown(
            '<div class="success-box">Enter your name and student ID above to start.</div>',
            unsafe_allow_html=True
        )
    else:
        st.divider()

        if "conversation" not in st.session_state:
            st.session_state.conversation = []
            st.session_state.turn_count = 0

        for msg in st.session_state.conversation:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><b>You</b><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            elif msg["role"] == "escalated":
                st.markdown(
                    f'<div class="chat-esc"><b>DRONA — Teacher Notified</b><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-ai"><b>DRONA</b><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

        doubt = st.chat_input("Type your doubt here...")

        if doubt:
            st.session_state.conversation.append({"role":"user","content":doubt})
            st.session_state.turn_count += 1

            with st.spinner("DRONA is thinking..."):
                try:
                    vs = load_vectorstore()
                    client = load_client()
                    docs = vs.similarity_search(doubt, k=3)
                    context = "\n\n".join([d.page_content for d in docs])
                    history = "\n".join([
                        f"{m['role'].upper()}: {m['content']}"
                        for m in st.session_state.conversation[-6:]
                    ])
                    subject = detect_subject(doubt)

                    if st.session_state.turn_count >= 4:
                        response = (
                            f"You have been working hard on this. "
                            f"I have notified your {subject} teacher. "
                            f"They will help you understand this concept better. Keep trying!"
                        )
                        st.session_state.conversation.append(
                            {"role":"escalated","content":response}
                        )
                        log_doubt(student_id, student_name, class_level,
                                 subject, doubt, response,
                                 st.session_state.turn_count, esc=1)
                        st.session_state.turn_count = 0
                    else:
                        response = get_ai_response(
                            client, doubt, context,
                            history, st.session_state.turn_count
                        )
                        st.session_state.conversation.append(
                            {"role":"assistant","content":response}
                        )
                        log_doubt(student_id, student_name, class_level,
                                 subject, doubt, response,
                                 st.session_state.turn_count)
                except Exception as e:
                    st.error(f"Error: {e}. Please try again.")

            st.rerun()

        if st.session_state.turn_count > 0:
            remaining = max(0, 4 - st.session_state.turn_count)
            st.markdown(
                f'<div class="turn-bar">Exchange {st.session_state.turn_count} of 4 — '
                f'{remaining} remaining before teacher notification</div>',
                unsafe_allow_html=True
            )

        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            st.session_state.turn_count = 0
            st.rerun()


elif page == "Teacher":
    st.markdown("#### Teacher Analytics Dashboard")
    st.caption("Real-time insights into student doubt patterns and learning gaps.")

    df = get_data()
    is_demo = df.empty
    if is_demo:
        df = demo_data()
        st.info("Showing demo data. Live data appears as students use the app.")

    total = len(df)
    resolved = int(df["resolved"].sum()) if "resolved" in df.columns else 0
    escalated = int(df["escalated"].sum()) if "escalated" in df.columns else 0
    rate = round(resolved/total*100, 1) if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        (total,"Total Doubts"),(resolved,"Resolved by AI"),
        (escalated,"Need Your Help"),(f"{rate}%","Resolution Rate")
    ]):
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Doubts by Subject**")
        subj = df["subject"].value_counts().reset_index()
        subj.columns = ["Subject", "Count"]
        fig = px.bar(
            subj, x="Subject", y="Count",
            color="Subject",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            height=260, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Resolution Breakdown**")
        res = pd.DataFrame({
            "Status": ["Resolved","Escalated","Pending"],
            "Count": [resolved, escalated, max(0,total-resolved-escalated)]
        })
        fig = px.pie(
            res, values="Count", names="Status", hole=0.5,
            color_discrete_sequence=["#38a169","#e53e3e","#ed8936"]
        )
        fig.update_layout(
            height=260, paper_bgcolor="white",
            margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Students Needing Attention**")
    if "escalated" in df.columns:
        at_risk = df[df["escalated"]==1][
            ["student_name","class_level","subject","doubt_text"]
        ].rename(columns={
            "student_name":"Student","class_level":"Class",
            "subject":"Subject","doubt_text":"Doubt"
        })
        if not at_risk.empty:
            st.dataframe(at_risk, use_container_width=True, height=180)
        else:
            st.markdown(
                '<div class="success-box">No students need escalation right now.</div>',
                unsafe_allow_html=True
            )

    if "subject" in df.columns and len(df) > 0:
        top = df["subject"].value_counts().index[0]
        cnt = df["subject"].value_counts().iloc[0]
        st.markdown(
            f'<div class="alert-box">Predictive Alert: {top} has {cnt} doubts '
            f'this period. Recommend a revision session.</div>',
            unsafe_allow_html=True
        )


elif page == "Admin":
    st.markdown("#### School-Wide Intelligence Dashboard")
    st.caption("Complete visibility into learning patterns across your school.")

    df = get_data()
    if df.empty:
        df = demo_data()
        st.info("Showing demo data. Live data appears as students use the app.")

    students = df["student_id"].nunique() if "student_id" in df.columns else 0
    esc_rate = round(df["escalated"].mean()*100,1) if "escalated" in df.columns else 0
    subj_count = df["subject"].nunique() if "subject" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        (students,"Active Students"),(len(df),"Total Doubts"),
        (f"{esc_rate}%","Escalation Rate"),(subj_count,"Subjects Active")
    ]):
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Subject Difficulty Heatmap**")
        if "class_level" in df.columns and "subject" in df.columns:
            heat = df.groupby(
                ["class_level","subject"]
            ).size().reset_index(name="doubts")
            pivot = heat.pivot(
                index="class_level",
                columns="subject",
                values="doubts"
            ).fillna(0)
            fig = px.imshow(
                pivot, color_continuous_scale="YlOrRd",
                labels={"color":"Doubts"}, text_auto=True
            )
            fig.update_layout(
                height=260, paper_bgcolor="white",
                margin=dict(t=10,b=10,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Doubt Volume Over Time**")
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            trend = df.groupby("date").size().reset_index(name="doubts")
            fig = px.area(
                trend, x="date", y="doubts",
                color_discrete_sequence=["#3182ce"]
            )
            fig.update_layout(
                height=260, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10,b=10,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    if "subject" in df.columns and "class_level" in df.columns and len(df) > 0:
        worst = df.groupby(["class_level","subject"]).size().idxmax()
        st.markdown(
            f'<div class="alert-box">Predictive Alert: {worst[0]} shows highest '
            f'difficulty in {worst[1]}. Recommend immediate teacher intervention.</div>',
            unsafe_allow_html=True
        )

    st.markdown("**Complete Doubt Log**")
    cols = ["student_name","class_level","subject","doubt_text","resolved","escalated"]
    avail = [c for c in cols if c in df.columns]
    st.dataframe(
        df[avail].rename(columns={
            "student_name":"Student","class_level":"Class",
            "subject":"Subject","doubt_text":"Doubt",
            "resolved":"Resolved","escalated":"Escalated"
        }),
        use_container_width=True, height=280
    )
