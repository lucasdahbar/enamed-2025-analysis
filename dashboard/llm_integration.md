# Integrating the explanation layer into the dashboard

Place `llm_explainer.py` next to `dashboard.py`. The explanation panel works
**with or without** a Gemini API key:

- **No key (default for reviewers):** a deterministic, offline grounded
  explanation is shown (`local_explanation`). No network, no secrets, fully
  reproducible.
- **With a key:** the text is generated live by Gemini (`explain`), as an
  optional enhancement. The reviewer never needs the authors' key.

## 1. Import (near the top of dashboard.py)

```python
try:
    from llm_explainer import AuditContext, explain, local_explanation
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
        live = explain(ctx)                      # None if no key / call fails
        text = live or local_explanation(ctx)    # offline fallback, no key needed
        st.markdown(text)
        if live:
            st.caption("Generated live with Gemini.")
        else:
            st.caption("Generated offline (deterministic, no API key needed). "
                       "Set GEMINI_API_KEY to enable the Gemini version.")
```

Adjust the variable names (`df_plot_local`, `df_course`, `selected_code`,
`course_score`, `national_median`, `performance_status`, `selected_lang`) to
match the dashboard.

## 3. Optional configuration (only to enable the live Gemini version)

```
pip install "google-genai>=2.0"
```
Provide a key via `.streamlit/secrets.toml` (`GEMINI_API_KEY = "..."`) or the
`GEMINI_API_KEY` environment variable. **Never commit a key to the repository.**
Without a key, the dashboard runs fully on the offline explanation.

## Quick diagnostic

```
GEMINI_API_KEY=... python llm_explainer.py    # prints the live result, or the exact failure reason
python llm_explainer.py                        # no key: prints the offline explanation
```
