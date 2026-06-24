import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



try:

    from llm_explainer import AuditContext, explain

    _LLM_AVAILABLE = True

except Exception:

    _LLM_AVAILABLE = False



st.set_page_config(

    page_title="ENAMED 2025 - Audit Hub",

    page_icon="🩺",

    layout="wide"

)



sns.set_theme(style="whitegrid")



LANG_PACK = {

    "English": {

        "sidebar_title": "🩺 ENAMED Audit Panel",

        "sidebar_desc": "Search and diagnose any medical program registered in the 2025 examination matrix.",

        "sidebar_select": "Select Target Course Code (CO_CURSO):",

        "sidebar_baseline": "National Baseline Context",

        "sidebar_median": "National Score Median",

        "sidebar_lgpd": "🔒 **LGPD & Compliance Note:** Institution names are masked via unique Course Codes (`CO_CURSO`) to ensure strict compliance with Brazilian Data Protection Regulations.",

        "main_title": "📊 Institutional Performance & Diagnostic Hub",

        "main_desc": "Deep dive analysis for Core Program Identifier:",

        "kpi_code": "Audited Program Code",

        "kpi_score": "Actual Exam Average Score",

        "kpi_delta": "vs Median",

        "kpi_tier": "Machine Learning Tier Class",

        "tier_high": "High Performance",

        "tier_low": "Standard / Low Performance",

        "chart_title": "💡 Feature Contribution Strength Mapping",

        "chart_desc": "*How much each institutional factor pushed this specific course inside the classification boundary.*",

        "chart_label_x": "Local Impact Force Score",

        "diag_title": "📋 Pedagogical Diagnostic & Action Items",

        "diag_desc": "*Automated operational summary extracted from survey metrics response weights.*",

        "strength_title": "🟢 Key Strengths",

        "strength_1": "**Solid Curricular Ingestion:** {mastery}% of the student body explicitly state they successfully studied and retained the core clinical evaluation framework (`I7_D`).",

        "strength_2": "**High Program Value:** Students widely validate that the course structure directly built their clinical baseline preparation (`I9_A`).",

        "friction_title": "🔴 Points of Improvement (Friction)",

        "friction_1": "**Curricular Friction Detected:** {gap}% of students highlighted 'unpreparedness / lack of content knowledge' (`I6_A`) as their primary bottleneck during the exam. **Action Item:** Review syllabus coverage latency.",

        "friction_no": "No critical student friction or content gap spikes detected for this institutional profile.",

        "friction_hard": "**Exam Anxiety / Difficulty Spike:** Over 20% evaluated the structure as exceptionally hard (`I1_D`), acting as a mathematical counterweight in model confidence."

    },

    "Português": {

        "sidebar_title": "🩺 Painel de Auditoria ENAMED",

        "sidebar_desc": "Pesquise e diagnostique qualquer curso de medicina registrado na matriz do exame de 2025.",

        "sidebar_select": "Selecione o Código do Curso (CO_CURSO):",

        "sidebar_baseline": "Contexto de Linha de Base Nacional",

        "sidebar_median": "Mediana Nacional da Nota",

        "sidebar_lgpd": "🔒 **Nota de Conformidade (LGPD):** Os nomes das instituições estão mascarados através dos Códigos de Curso (`CO_CURSO`) para garantir estrito cumprimento com a Lei Geral de Proteção de Dados.",

        "main_title": "📊 Hub de Diagnóstico & Desempenho Institucional",

        "main_desc": "Análise aprofundada para o Identificador do Programa:",

        "kpi_code": "Código do Curso Auditado",

        "kpi_score": "Nota Média Real no Exame",

        "kpi_delta": "vs Mediana",

        "kpi_tier": "Classe de Desempenho (M. Learning)",

        "tier_high": "Alta Performance",

        "tier_low": "Desempenho Padrão / Baixo",

        "chart_title": "💡 Mapeamento de Força de Contribuição de Atributos",

        "chart_desc": "*O quanto cada fator institucional empurrou este curso específico dentro da fronteira de classificação.*",

        "chart_label_x": "Pontuação de Força de Impacto Local",

        "diag_title": "📋 Diagnóstico Pedagógico & Planos de Ação",

        "diag_desc": "*Resumo operacional automatizado extraído dos pesos de resposta do questionário.*",

        "strength_title": "🟢 Pontos Fortes",

        "strength_1": "**Sólida Absorção Curricular:** {mastery}% do corpo discente afirma explicitamente que estudou e aprendeu a maior parte do conteúdo avaliado (`I7_D`).",

        "strength_2": "**Alto Valor do Programa:** Os alunos validam amplamente que a estrutura do curso contribuiu muito para a preparação da sua base clínica (`I9_A`).",

        "friction_title": "🔴 Pontos de Melhoria (Atrito)",

        "friction_1": "**Atrito Curricular Detectado:** {gap}% dos alunos destacaram o 'desconhecimento de conteúdo' (`I6_A`) como o principal gargalo durante a prova. **Plano de Ação:** Revisar a latência de cobertura da ementa.",

        "friction_no": "Nenhum atrito crítico ou pico de desconhecimento de conteúdo foi detectado para este perfil institucional.",

        "friction_hard": "**Pico de Ansiedade / Dificuldade na Prova:** Mais de 20% avaliaram a estrutura como excepcionalmente difícil (`I1_D`), atuando como um contrapeso matemático.",

    }

}





# Mapping translations explicitly to match raw variables dynamically

FEATURE_TRANSLATION = {

    "English": {

        'pct_CO_RS_I1_C': 'Exam Difficulty: Medium (I1_C)',

        'pct_CO_RS_I7_D': 'High Content Mastery (I7_D)',

        'pct_CO_RS_I4_B': 'Clear Question Stems (I4_B)',

        'pct_CO_RS_I9_A': 'High Course Contribution (I9_A)',

        'pct_CO_RS_I6_A': 'Friction: Content Gaps (I6_A)',

        'pct_CO_RS_I3_C': 'Adequate Exam Length (I3_C)',

        'pct_CO_RS_I1_D': 'Friction: Exam Considered Hard (I1_D)'

    },

    "Português": {

        'pct_CO_RS_I1_C': 'Dificuldade da Prova: Média (I1_C)',

        'pct_CO_RS_I7_D': 'Alta Retenção de Conteúdo (I7_D)',

        'pct_CO_RS_I4_B': 'Enunciados Claros (I4_B)',

        'pct_CO_RS_I9_A': 'Alta Contribuição do Curso (I9_A)',

        'pct_CO_RS_I6_A': 'Atrito: Desconhecimento (I6_A)',

        'pct_CO_RS_I3_C': 'Extensão de Prova Adequada (I3_C)',

        'pct_CO_RS_I1_D': 'Atrito: Prova Considerada Difícil (I1_D)'

    }

}



@st.cache_data

def load_audit_data():

    np.random.seed(42)

    courses = [13103, 5001167, 10020, 20045, 30067, 40089]

    data = {

        'CO_CURSO': courses,

        'avg_score_general': [68.5, 82.4, 45.2, 71.0, 52.3, 61.2],

        'pct_CO_RS_I1_C': [60, 85, 30, 65, 40, 55],

        'pct_CO_RS_I7_D': [75, 90, 40, 80, 50, 70],

        'pct_CO_RS_I4_B': [68, 88, 35, 72, 45, 60],

        'pct_CO_RS_I9_A': [80, 95, 50, 85, 55, 75],

        'pct_CO_RS_I6_A': [25, 5, 55, 12, 48, 22],

        'pct_CO_RS_I1_D': [22, 8, 45, 15, 38, 18],

        'pct_CO_RS_I3_C': [70, 92, 42, 78, 52, 66]

    }

    df = pd.DataFrame(data)

    median_score = df['avg_score_general'].median()

    df['performance_class'] = np.where(df['avg_score_general'] >= median_score, 1, 0)

    return df, median_score



df_ml, national_median = load_audit_data()



features_selected = ['pct_CO_RS_I1_C', 'pct_CO_RS_I7_D', 'pct_CO_RS_I4_B', 'pct_CO_RS_I9_A', 'pct_CO_RS_I6_A', 'pct_CO_RS_I3_C', 'pct_CO_RS_I1_D']

mock_importances = np.array([0.25, 0.22, 0.18, 0.15, 0.10, 0.06, 0.04])



st.sidebar.title("🌐 Language / Idioma")

selected_lang = st.sidebar.selectbox("Choose Language:", ["English", "Português"])



texts = LANG_PACK[selected_lang]



st.sidebar.markdown("---")

st.sidebar.title(texts["sidebar_title"])

st.sidebar.write(texts["sidebar_desc"])



all_codes = sorted(df_ml['CO_CURSO'].unique())

selected_code = st.sidebar.selectbox(

    texts["sidebar_select"],

    all_codes,

    index=all_codes.index(13103) if 13103 in all_codes else 0

)



st.sidebar.markdown("---")

st.sidebar.markdown(f"### {texts['sidebar_baseline']}")

st.sidebar.metric(label=texts["sidebar_median"], value=f"{national_median:.2f}")



st.sidebar.markdown("---")

st.sidebar.caption(texts["sidebar_lgpd"])



st.title(texts["main_title"])

st.write(f"{texts['main_desc']} **Course {selected_code}**")

st.markdown("---")



df_course = df_ml[df_ml['CO_CURSO'] == selected_code]

course_score = df_course['avg_score_general'].values[0]

is_high = df_course['performance_class'].values[0] == 1

performance_status = texts["tier_high"] if is_high else texts["tier_low"]



kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

kpi_col1.metric(label=texts["kpi_code"], value=str(selected_code))

kpi_col2.metric(label=texts["kpi_score"], value=f"{course_score:.2f}", delta=f"{course_score - national_median:.2f} {texts['kpi_delta']}")

kpi_col3.metric(label=texts["kpi_tier"], value=performance_status)





st.markdown("---")



left_col, right_col = st.columns([3, 2])



with left_col:

    st.subheader(texts["chart_title"])

    st.write(texts["chart_desc"])

   

    local_impact = df_course[features_selected].values[0] * mock_importances

   

    # Building the mapping dataframe dynamically before sorting

    df_plot_local = pd.DataFrame({

        'Feature_Raw': features_selected,

        'Contribution_Strength': local_impact

    })

   

    # Mapping raw columns to translated labels based on selected language

    df_plot_local['Feature/Driver'] = df_plot_local['Feature_Raw'].map(FEATURE_TRANSLATION[selected_lang])

   

    # Sorting ensures both the bars and the Y-axis labels change dynamically per course

    df_plot_local = df_plot_local.sort_values(by='Contribution_Strength', ascending=False)

   

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(x='Contribution_Strength', y='Feature/Driver', data=df_plot_local, palette='viridis', ax=ax)

    ax.set_xlabel(texts["chart_label_x"])

    ax.set_ylabel('')

    st.pyplot(fig)

    plt.close(fig)



with right_col:

    st.subheader(texts["diag_title"])

    st.write(texts["diag_desc"])

   

    content_gap_pct = df_course['pct_CO_RS_I6_A'].values[0]

    mastery_pct = df_course['pct_CO_RS_I7_D'].values[0]

   

    st.markdown(f"#### {texts['strength_title']}")

    st.success(texts["strength_1"].format(mastery=mastery_pct))

   

    if df_course['pct_CO_RS_I9_A'].values[0] > 70:

        st.success(texts["strength_2"])

       

    st.markdown(f"#### {texts['friction_title']}")

    if content_gap_pct > 20:

        st.error(texts["friction_1"].format(gap=content_gap_pct))

    else:

        st.info(texts["friction_no"])

       

    if df_course['pct_CO_RS_I1_D'].values[0] > 20:

        st.warning(texts["friction_hard"])



        st.markdown("---")

_label = "Explanation (Gemini)" if selected_lang == "English" else "Explicação (Gemini)"

with st.expander(_label, expanded=False):

    if not _LLM_AVAILABLE:

        st.info("llm_explainer.py was not found next to dashboard.py.")

    else:

        _friction = {"pct_CO_RS_I6_A", "pct_CO_RS_I1_D"}

        _strength = {"pct_CO_RS_I7_D", "pct_CO_RS_I9_A", "pct_CO_RS_I4_B"}

        contribs = []

        for _, row in df_plot_local.iterrows():

            raw = row["Feature_Raw"]

            sentiment = "friction" if raw in _friction else ("strength" if raw in _strength else "neutral")

            contribs.append({

                "feature": raw.replace("pct_CO_RS_", ""),

                "label": row["Feature/Driver"],

                "value_pct": float(df_course[raw].values[0]),

                "impact": round(float(row["Contribution_Strength"]), 2),

                "sentiment": sentiment,

            })

        ctx = AuditContext(

            course_code=int(selected_code),

            avg_score=float(course_score),

            national_median=float(national_median),

            predicted_tier=performance_status,

            contributions=contribs,

            language=selected_lang,

            confidence_pct=None,   # pass model.predict_proba(...) here when available

        )

        with st.spinner("..."):

            text = explain(ctx)

        if text:

            st.markdown(text)

        else:

            st.info("Set GEMINI_API_KEY in .streamlit/secrets.toml to enable explanations. "

                    "The rule-based diagnostic above remains valid.")