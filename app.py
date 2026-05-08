import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PragyanAI Quiz Performance Analytics Platform",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .weak-topic  { color: #ff4b4b; font-weight: bold; }
    .strong-topic{ color: #09ab3b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PRESCRIBED DATASET URL (from project spec)
# ─────────────────────────────────────────────
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/pragyanaischool/"
    "VTU_Internship_DataSets/refs/heads/main/"
    "student_data_performance_analysis_project_9.csv"
)

# ─────────────────────────────────────────────
# LOAD DATA  (cached for performance)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    try:
        df = pd.read_csv(file) if file is not None else pd.read_csv(DEFAULT_DATA_URL)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
    return df

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🧠 PragyanAI Menu")
st.sidebar.markdown("---")

st.sidebar.subheader("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file", type=["csv"],
    help="Upload student quiz CSV. If skipped, the prescribed dataset is loaded."
)

if uploaded_file is not None:
    st.sidebar.success(f"✅ Using: {uploaded_file.name}")
    df = load_data(file=uploaded_file)
else:
    st.sidebar.info("ℹ️ Using prescribed dataset (GitHub)")
    df = load_data()

if df.empty:
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("⬇️ Download Data")
st.sidebar.download_button(
    label="Download Current Dataset",
    data=df.to_csv(index=False),
    file_name="student_data_export.csv",
    mime="text/csv"
)
st.sidebar.markdown("---")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
if 'Attempts' in df.columns and 'Attempts_Count' not in df.columns:
    df = df.rename(columns={'Attempts': 'Attempts_Count'})

if 'Test_Cases_Passed' in df.columns:
    if df['Test_Cases_Passed'].dtype == 'O':
        df['Test_Cases_Passed'] = (
            df['Test_Cases_Passed'].astype(str)
            .str.replace('%', '').astype(float) / 100.0
        )
    elif df['Test_Cases_Passed'].max() > 1.0:
        df['Test_Cases_Passed'] = df['Test_Cases_Passed'] / 100.0

for col in ['Correct', 'Time_Taken', 'Hint_Used', 'Attempted',
            'Concept_Understood', 'Attempts_Count', 'CGPA']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ─────────────────────────────────────────────
# DERIVED METRICS  (from spec Section 6)
# ─────────────────────────────────────────────
# 1. Weakness Score = 1 − Accuracy
df['Weakness_Score'] = 1 - df['Correct']

# 2. Concept Mastery = Correct × (1 − Hint) × TestCases
hint = df['Hint_Used']           if 'Hint_Used'           in df.columns else 0
tcp  = df['Test_Cases_Passed']   if 'Test_Cases_Passed'   in df.columns else 1
df['Concept_Mastery'] = df['Correct'] * (1 - hint) * tcp

# 3. Learning Efficiency = Accuracy / Time  (avoid div/0)
df['Learning_Efficiency'] = (
    df['Correct'] * 100 / (df['Time_Taken'] + 1)
    if 'Time_Taken' in df.columns else 0
)

# ─────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────
page = st.sidebar.radio("Navigation", [
    "📊 Global Dashboard",
    "🔍 Deep Analysis",
    "📈 Advanced Analytics",
    "🤖 AI Tutor Panel",
    "🏫 Faculty Panel"
])

# ══════════════════════════════════════════════
# PAGE 1 — GLOBAL DASHBOARD
# ══════════════════════════════════════════════
if page == "📊 Global Dashboard":
    st.title("📊 Global Dashboard: Performance Overview")
    st.markdown("Track student performance across topics, departments, and college tiers.")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    total_students = df['Student_ID'].nunique() if 'Student_ID' in df.columns else len(df)
    avg_accuracy   = df['Correct'].mean() * 100
    total_attempts = df['Attempts_Count'].sum() if 'Attempts_Count' in df.columns else len(df)
    avg_mastery    = df['Concept_Mastery'].mean() * 100

    c1.metric("🎓 Total Students",      f"{total_students:,}")
    c2.metric("🎯 Overall Accuracy",    f"{avg_accuracy:.1f}%")
    c3.metric("📝 Total Attempts",      f"{total_attempts:,.0f}")
    c4.metric("🧠 Avg Concept Mastery", f"{avg_mastery:.1f}%")

    st.markdown("---")

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("🔴 Top 5 Weakest Topics")
        wt = df.groupby('Topic')['Correct'].mean().sort_values().head(5) * 100
        fig = px.bar(wt, orientation='h', color=wt.values,
                     color_continuous_scale="Reds_r",
                     labels={'value': 'Accuracy (%)', 'index': 'Topic'},
                     title="Weakest Topics by Accuracy")
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.subheader("🏢 Department-wise Performance")
        dp = df.groupby('Department')['Correct'].mean().reset_index()
        dp['Accuracy (%)'] = dp['Correct'] * 100
        fig = px.pie(dp, names='Department', values='Accuracy (%)',
                     hole=0.4, title="Accuracy by Department")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏫 College Tier Comparison")
    tp = df.groupby('College_Tier')['Correct'].mean().reset_index().sort_values('College_Tier')
    tp['Accuracy (%)'] = tp['Correct'] * 100
    fig = px.bar(tp, x='College_Tier', y='Accuracy (%)', color='College_Tier',
                 text='Accuracy (%)', title="Accuracy by College Tier")
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Student Ranking (Platform Feature — Dashboard)
    st.subheader("🏆 Student Ranking — Top 10 by Concept Mastery")
    if 'Student_ID' in df.columns:
        rank_df = df.groupby('Student_ID').agg(
            Accuracy        = ('Correct',          'mean'),
            Concept_Mastery = ('Concept_Mastery',  'mean'),
            Total_Attempts  = ('Correct',          'count')
        ).reset_index()
        rank_df['Accuracy (%)']        = (rank_df['Accuracy']        * 100).round(1)
        rank_df['Concept Mastery (%)'] = (rank_df['Concept_Mastery'] * 100).round(1)
        rank_df = rank_df.sort_values('Concept_Mastery', ascending=False).head(10)
        rank_df.insert(0, 'Rank', range(1, len(rank_df) + 1))
        st.dataframe(
            rank_df[['Rank', 'Student_ID', 'Accuracy (%)', 'Concept Mastery (%)', 'Total_Attempts']],
            use_container_width=True, hide_index=True
        )

# ══════════════════════════════════════════════
# PAGE 2 — DEEP ANALYSIS  (all 9 analyses)
# ══════════════════════════════════════════════
elif page == "🔍 Deep Analysis":
    st.title("🔍 Deep Analysis (Multi-Layer)")
    st.markdown("9 analysis layers: Topic → Subtopic → Difficulty → Dept → College → Time → Hints → Attempts → Test Cases")

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📚 Topic & Subtopic",
        "🎯 Difficulty",
        "⏱ Time & Attempts",
        "💡 Hints & Learning Signals",
        "🏢 Dept & College",
        "🗺 Cross-Analysis Heatmap"
    ])

    # ── Analysis 1 & 2: Topic & Subtopic ──
    with t1:
        st.subheader("1. Topic-wise Accuracy")
        ta = df.groupby('Topic')['Correct'].mean().reset_index().sort_values('Correct', ascending=False)
        ta['Accuracy (%)'] = (ta['Correct'] * 100).round(1)
        fig = px.bar(ta, x='Topic', y='Accuracy (%)', color='Accuracy (%)',
                     color_continuous_scale='RdYlGn',
                     title="Topic-wise Accuracy (Green = Strong / Red = Weak)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("2. Subtopic Analysis")
        sel_topic = st.selectbox("Select Topic", ta['Topic'].unique())
        sa = df[df['Topic'] == sel_topic].groupby('Subtopic')['Correct'].mean().reset_index()
        sa['Accuracy (%)'] = (sa['Correct'] * 100).round(1)
        fig = px.bar(sa, x='Subtopic', y='Accuracy (%)', color='Accuracy (%)',
                     color_continuous_scale='RdYlGn',
                     title=f"Subtopic Accuracy: {sel_topic}")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Students fail in complexity (nested/edge cases), not in basics.")

    # ── Analysis 3: Difficulty ──
    with t2:
        st.subheader("3. Difficulty Analysis — Performance Curve")
        diff_order = ["Easy", "Medium", "Hard"]
        da = df.groupby('Difficulty')['Correct'].mean().reset_index()
        da['Difficulty'] = pd.Categorical(da['Difficulty'], categories=diff_order, ordered=True)
        da = da.sort_values('Difficulty')
        da['Accuracy (%)'] = (da['Correct'] * 100).round(1)

        fig = px.line(da, x='Difficulty', y='Accuracy (%)', markers=True,
                      title="Difficulty Curve: Easy → Medium → Hard")
        fig.update_traces(line=dict(color='red', width=4), marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(da, x='Difficulty', y='Accuracy (%)',
                      color='Difficulty', text='Accuracy (%)',
                      color_discrete_map={'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'})
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
        st.info("💡 Learning gap in problem solving — Medium difficulty is the real learning zone.")

    # ── Analysis 6 & 8: Time & Attempts ──
    with t3:
        st.subheader("6. Time Taken Analysis")
        if 'Time_Taken' in df.columns:
            df['Time_Bin'] = pd.cut(
                df['Time_Taken'], bins=[0, 30, 120, float('inf')],
                labels=['<30 sec (Too Fast)', '30–120 sec (Optimal)', '>120 sec (Confused)']
            )
            tma = df.groupby('Time_Bin')['Correct'].mean().reset_index()
            tma['Accuracy (%)'] = tma['Correct'] * 100
            fig = px.bar(tma, x='Time_Bin', y='Accuracy (%)', color='Time_Bin',
                         title="Accuracy by Time Taken")
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Too fast = guessing | 30–120s = best zone | >120s = confusion")

        st.subheader("8. Attempts Analysis")
        if 'Attempts_Count' in df.columns:
            df['Attempts_Bin'] = pd.cut(
                df['Attempts_Count'], bins=[0, 1, 2, float('inf')],
                labels=['1 Attempt', '2 Attempts', '3+ Attempts']
            )
            aa = df.groupby('Attempts_Bin')['Correct'].mean().reset_index()
            aa['Accuracy (%)'] = aa['Correct'] * 100
            fig = px.bar(aa, x='Attempts_Bin', y='Accuracy (%)', color='Attempts_Bin',
                         title="Accuracy by Attempt Count")
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 2 attempts = highest accuracy (learning) | 3+ = guessing pattern")

        if 'Attempted' in df.columns:
            st.subheader("Attempted vs Not Attempted")
            ac = df['Attempted'].value_counts().reset_index()
            ac.columns = ['Attempted', 'Count']
            ac['Attempted'] = ac['Attempted'].map({1: 'Attempted', 0: 'Not Attempted'})
            fig = px.pie(ac, names='Attempted', values='Count', hole=0.4,
                         title="Question Attempt Rate")
            st.plotly_chart(fig, use_container_width=True)

    # ── Analysis 7, 9 + Concept_Understood, Code_Submission ──
    with t4:
        st.subheader("7. Hint Usage Analysis")
        if 'Hint_Used' in df.columns:
            ha = df.groupby('Hint_Used')['Correct'].mean().reset_index()
            ha['Label']        = ha['Hint_Used'].map({0: 'No Hint', 1: 'Hint Used'})
            ha['Accuracy (%)'] = ha['Correct'] * 100
            fig = px.bar(ha, x='Label', y='Accuracy (%)', color='Label',
                         color_discrete_map={'No Hint': '#2ecc71', 'Hint Used': '#e74c3c'},
                         title="Impact of Hint Usage on Accuracy")
            st.plotly_chart(fig, use_container_width=True)
            st.warning("💡 Over-reliance on hints — students with no hints score higher.")

        st.subheader("9. Test Case Analysis (Logic vs Syntax)")
        if 'Test_Cases_Passed' in df.columns:
            df['TC_Bin'] = pd.cut(
                df['Test_Cases_Passed'], bins=[-0.1, 0.49, 0.80, 1.0],
                labels=['<50% (Weak Logic)', '50–80% (Medium)', '>80% (Strong Coder)']
            )
            tca = df.groupby('TC_Bin')['Correct'].mean().reset_index()
            tca['Accuracy (%)'] = tca['Correct'] * 100
            fig = px.bar(tca, x='TC_Bin', y='Accuracy (%)', color='TC_Bin',
                         title="Test Cases Passed vs Quiz Accuracy")
            st.plotly_chart(fig, use_container_width=True)

        if 'Concept_Understood' in df.columns:
            st.subheader("Concept Understood vs Accuracy")
            cu = df.groupby('Concept_Understood')['Correct'].mean().reset_index()
            cu['Label']        = cu['Concept_Understood'].map({0: 'Not Understood', 1: 'Understood'})
            cu['Accuracy (%)'] = cu['Correct'] * 100
            fig = px.bar(cu, x='Label', y='Accuracy (%)', color='Label',
                         color_discrete_map={'Understood': '#2ecc71', 'Not Understood': '#e74c3c'},
                         title="Concept Understood → Accuracy Impact")
            st.plotly_chart(fig, use_container_width=True)

        if 'Code_Submission' in df.columns:
            st.subheader("Code Submission vs Accuracy")
            cs = df.groupby('Code_Submission')['Correct'].mean().reset_index()
            cs['Accuracy (%)'] = cs['Correct'] * 100
            fig = px.bar(cs, x='Code_Submission', y='Accuracy (%)',
                         color='Code_Submission', title="Code Submission Impact on Accuracy")
            st.plotly_chart(fig, use_container_width=True)

    # ── Analysis 4 & 5: Dept & College + CGPA + Batch ──
    with t5:
        st.subheader("4. Department-wise Analysis")
        dpa = df.groupby('Department')['Correct'].mean().reset_index().sort_values('Correct')
        dpa['Accuracy (%)'] = (dpa['Correct'] * 100).round(1)
        fig = px.bar(dpa, x='Accuracy (%)', y='Department', orientation='h',
                     color='Accuracy (%)', color_continuous_scale='RdYlGn',
                     title="Department-wise Accuracy")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Non-tech students (Mechanical, Civil) struggle heavily — need Foundation Track.")

        st.subheader("5. College Tier Analysis")
        cta = df.groupby('College_Tier')['Correct'].mean().reset_index().sort_values('College_Tier')
        cta['Accuracy (%)'] = (cta['Correct'] * 100).round(1)
        fig = px.bar(cta, x='College_Tier', y='Accuracy (%)', color='College_Tier',
                     text='Accuracy (%)', title="College Tier vs Accuracy")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Skill gap visible: Tier 1 > Tier 2 > Tier 3 > Tier 4")

        if 'CGPA' in df.columns:
            st.subheader("CGPA Band vs Quiz Accuracy")
            df['CGPA_Bin'] = pd.cut(df['CGPA'], bins=[0, 5, 6, 7, 8, 10],
                                    labels=['<5', '5–6', '6–7', '7–8', '8+'])
            cg = df.groupby('CGPA_Bin')['Correct'].mean().reset_index()
            cg['Accuracy (%)'] = (cg['Correct'] * 100).round(1)
            fig = px.bar(cg, x='CGPA_Bin', y='Accuracy (%)', color='Accuracy (%)',
                         color_continuous_scale='Blues', title="CGPA Band vs Accuracy")
            st.plotly_chart(fig, use_container_width=True)

        if 'Batch' in df.columns:
            st.subheader("Batch-wise Accuracy Trend")
            ba = df.groupby('Batch')['Correct'].mean().reset_index()
            ba['Accuracy (%)'] = (ba['Correct'] * 100).round(1)
            fig = px.line(ba, x='Batch', y='Accuracy (%)', markers=True,
                          title="Batch-wise Accuracy Trend")
            st.plotly_chart(fig, use_container_width=True)

    # ── Cross-Analysis Heatmaps ──
    with t6:
        st.subheader("Heatmap: Topic × Department Accuracy")
        hd = df.groupby(['Topic', 'Department'])['Correct'].mean().unstack() * 100
        fig = px.imshow(hd, text_auto=".1f", aspect="auto",
                        color_continuous_scale="RdYlGn",
                        title="Topic vs Department Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)

        if 'College_Tier' in df.columns:
            st.subheader("Heatmap: Topic × College Tier Accuracy")
            ht = df.groupby(['Topic', 'College_Tier'])['Correct'].mean().unstack() * 100
            fig = px.imshow(ht, text_auto=".1f", aspect="auto",
                            color_continuous_scale="Blues",
                            title="Topic vs College Tier Accuracy (%)")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 3 — ADVANCED ANALYTICS
# ══════════════════════════════════════════════
elif page == "📈 Advanced Analytics":
    st.title("📈 Advanced Analytics & Metrics")
    st.markdown("4 advanced computed metrics + Skill Map + Key Insights (GOLD 🏅)")

    # 4 Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    easy_acc = df[df['Difficulty'] == 'Easy']['Correct'].mean() if 'Difficulty' in df.columns else 0
    hard_acc = df[df['Difficulty'] == 'Hard']['Correct'].mean() if 'Difficulty' in df.columns else 0

    c1.metric("⚠️ Avg Weakness Score",
              f"{df['Weakness_Score'].mean():.2f}",
              help="Weakness = 1 − Accuracy")
    c2.metric("🧠 Avg Concept Mastery",
              f"{df['Concept_Mastery'].mean()*100:.1f}%",
              help="Mastery = Correct × (1−Hint) × TestCases")
    c3.metric("📉 Difficulty Gap (Easy−Hard)",
              f"{(easy_acc - hard_acc)*100:.1f}%",
              help="Gap = Easy Accuracy − Hard Accuracy")
    c4.metric("⚡ Avg Learning Efficiency",
              f"{df['Learning_Efficiency'].mean():.2f}",
              help="Efficiency = Accuracy / Time")

    st.markdown("---")

    # Skill Map
    st.markdown("### 🗺️ Skill Map: Accuracy vs Mastery vs Efficiency")
    ts = df.groupby('Topic').agg(
        Accuracy        = ('Correct',           'mean'),
        Concept_Mastery = ('Concept_Mastery',   'mean'),
        Efficiency      = ('Learning_Efficiency','mean'),
        Weakness        = ('Weakness_Score',    'mean')
    ).reset_index()
    ts['Accuracy (%)'] = (ts['Accuracy'] * 100).round(1)
    ts['Mastery (%)']  = (ts['Concept_Mastery'] * 100).round(1)

    fig = px.scatter(ts, x='Accuracy (%)', y='Mastery (%)',
                     color='Efficiency', size='Weakness',
                     hover_data=['Topic'], text='Topic',
                     color_continuous_scale='RdYlGn',
                     title="Skill Map — Bubble size = Weakness | Color = Efficiency")
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    # Weakness Ranking
    st.markdown("### ⚠️ Topic Weakness Score Ranking")
    wr = ts.sort_values('Weakness', ascending=False)
    fig = px.bar(wr, x='Topic', y='Weakness', color='Weakness',
                 color_continuous_scale='Reds',
                 title="Topic Weakness Score (Higher = Needs More Attention)")
    st.plotly_chart(fig, use_container_width=True)

    # KEY INSIGHTS (GOLD)
    st.markdown("---")
    st.markdown("### 💡 Key Insights (GOLD 🏅)")
    ca, cb = st.columns(2)
    with ca:
        st.success ("✅ **Insight 1:** Students understand basics but fail in complexity.")
        st.error   ("❌ **Insight 2:** Recursion, OOP, and Problem Solving are the weakest topics.")
        st.warning ("⚠️ **Insight 3:** Non-tech students need a Foundation Track before Module 4.")
    with cb:
        st.info    ("📌 **Insight 4:** Coding ability (Test Cases) ≠ Theoretical understanding (Quiz).")
        st.success ("✅ **Insight 5:** Medium difficulty is the optimal real learning zone.")

# ══════════════════════════════════════════════
# PAGE 4 — AI TUTOR PANEL
# ══════════════════════════════════════════════
elif page == "🤖 AI Tutor Panel":
    st.title("🤖 Personalized AI Tutor")
    st.markdown("Identify student gaps → Suggest topics → Recommend quizzes.")

    if 'Student_ID' not in df.columns:
        st.error("Student_ID column not found.")
        st.stop()

    student_id   = st.selectbox("🎓 Select Student ID", sorted(df['Student_ID'].unique()))
    student_data = df[df['Student_ID'] == student_id]

    st.subheader(f"📋 Profile: Student {student_id}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",         f"{student_data['Correct'].mean()*100:.1f}%")
    c2.metric("Concept Mastery",  f"{student_data['Concept_Mastery'].mean()*100:.1f}%")
    c3.metric("Quizzes Attempted", len(student_data))
    if 'CGPA' in student_data.columns:
        c4.metric("CGPA", f"{student_data['CGPA'].mean():.2f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🔴 Weak Areas Detected")
        weak_areas = student_data[student_data['Correct'] == 0]['Topic'].value_counts()
        if not weak_areas.empty:
            for topic, count in weak_areas.items():
                st.markdown(
                    f"- <span class='weak-topic'>{topic}</span> ({count} failed attempts)",
                    unsafe_allow_html=True
                )
        else:
            st.success("No major weak areas detected!")

        st.markdown("### 📊 Student Topic Performance")
        st_tp = student_data.groupby('Topic')['Correct'].mean().reset_index()
        st_tp['Accuracy (%)'] = (st_tp['Correct'] * 100).round(1)
        fig = px.bar(st_tp, x='Topic', y='Accuracy (%)', color='Accuracy (%)',
                     color_continuous_scale='RdYlGn',
                     title=f"Topic Accuracy — Student {student_id}")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### 🎯 AI Recommendations")
        if not weak_areas.empty:
            top_w  = weak_areas.index[0]
            sec_w  = weak_areas.index[1] if len(weak_areas) > 1 else None
            st.info   (f"👉 **Focus Area:** Start extra modules on **{top_w}**.")
            st.warning(f"👉 **Practice Path:** Attempt 5 Easy questions in **{top_w}** → then Medium.")
            if sec_w:
                st.info(f"👉 **Next Priority:** After {top_w}, tackle **{sec_w}**.")
            st.success("👉 **Hint Strategy:** Avoid hints — solve independently for real mastery.")
        else:
            st.success("🏆 Ready for competitive programming tracks!")
            st.info("👉 Challenge yourself with Hard difficulty questions.")

        st.markdown("### 📚 Recommended Quiz Topics")
        topics_to_recommend = weak_areas.index[:3] if not weak_areas.empty else []
        if len(topics_to_recommend) > 0:
            for i, topic in enumerate(topics_to_recommend, 1):
                st.markdown(f"**{i}.** 📖 Practice `{topic}` — Easy → Medium → Hard progression")
        else:
            st.info("No specific topic recommendations. Keep up the great work!")

# ══════════════════════════════════════════════
# PAGE 5 — FACULTY PANEL
# ══════════════════════════════════════════════
elif page == "🏫 Faculty Panel":
    st.title("🏫 Faculty & Curriculum Improvement Panel")
    st.markdown("Data-driven insights to improve curriculum and teaching methods.")

    tm, td, tc = st.tabs(["📦 Module Diagnostics", "🏢 Department View", "🛠 Curriculum Plan"])

    with tm:
        st.subheader("Module-level Diagnostics")
        if 'Module' in df.columns:
            ma = df.groupby('Module')['Correct'].mean().reset_index().sort_values('Correct')
            ma['Accuracy (%)'] = (ma['Correct'] * 100).round(1)
            fig = px.bar(ma, x='Accuracy (%)', y='Module', orientation='h',
                         color='Accuracy (%)', color_continuous_scale='Reds_r',
                         text='Accuracy (%)', title="Module Accuracy — Areas to Re-Teach")
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            sel_mod = st.selectbox("Select Module → See Topic Breakdown",
                                   sorted(df['Module'].unique()))
            mt = df[df['Module'] == sel_mod].groupby('Topic')['Correct'].mean().reset_index()
            mt['Accuracy (%)'] = (mt['Correct'] * 100).round(1)
            fig = px.bar(mt, x='Topic', y='Accuracy (%)', color='Accuracy (%)',
                         color_continuous_scale='RdYlGn',
                         title=f"Topic Accuracy within {sel_mod}")
            st.plotly_chart(fig, use_container_width=True)

    with td:
        st.subheader("Department × Topic Weakness Matrix")
        if 'Department' in df.columns and 'Topic' in df.columns:
            dt = df.groupby(['Department', 'Topic'])['Correct'].mean().unstack() * 100
            fig = px.imshow(dt, text_auto=".1f", aspect="auto",
                            color_continuous_scale="RdYlGn",
                            title="Department × Topic Accuracy Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    with tc:
        st.markdown("### 🛠 Curriculum Action Plan")

        st.error(
            "**1. 🔴 Fix Weak Areas**\n"
            "- Add more recursion and OOP practice problems\n"
            "- Add visual explanations instead of just code\n"
            "- Schedule extra sessions for weakest modules"
        )
        st.warning(
            "**2. 🟡 Adaptive Learning Integration**\n"
            "- Follow Easy → Medium → Hard progression strictly\n"
            "- Weak topic detected → assign extra quiz modules automatically"
        )
        st.info(
            "**3. 🔵 Core Branch Targeting**\n"
            "- Mechanical/Civil + Tier 3/4 → 'Foundation Track' bootcamp before Module 4\n"
            "- Non-tech students need Python basics intensive course first"
        )
        st.success(
            "**4. 🟢 Personalized Learning**\n"
            "- Weak topic identified → assign personalised learning path\n"
            "- AI Tutor recommends quizzes based on student gap analysis"
        )

        st.markdown("---")
        st.markdown("""
### 📌 Final Verdict

| ❌ Old Approach | ✅ New Approach |
|---|---|
| Teaching same for all | Teaching based on data-driven weaknesses |
| Fixed curriculum | Adaptive Easy → Medium → Hard |
| No personalization | AI-driven topic recommendations |
| Ignoring tier/dept gap | Tier-specific foundation tracks |

---
### 🚀 What This Platform Does
- 🔍 **Tracks concepts** per student, topic, and subtopic
- 🎯 **Identifies gaps** across departments and college tiers
- 📈 **Improves curriculum** based on real data
- 🤖 **Personalizes learning** with AI Tutor recommendations
        """)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("PragyanAI Learning Intelligence Engine v2.0")
