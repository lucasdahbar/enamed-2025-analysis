# Integrating the explanation layer into the dashboard

Place `llm_explainer.py` next to `dashboard.py`.

## 1. Import (near the top of dashboard.py)

```python
try:
    from llm_explainer import AuditContext, explain
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False
```

## 2. Render the explanation (after the diagnostics section)

```python
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
```

Adjust the variable names (`df_plot_local`, `df_course`, `selected_code`, `course_score`,
`national_median`, `performance_status`, `selected_lang`) to match the dashboard.

## 3. Configuration

```
pip install google-genai
```
Provide a key via `.streamlit/secrets.toml`:
```
GEMINI_API_KEY = "your-key"
```
or the environment variable `GEMINI_API_KEY`. Without a key, the dashboard keeps its
rule-based diagnostic. Only masked, course-level aggregates are sent to the service.
