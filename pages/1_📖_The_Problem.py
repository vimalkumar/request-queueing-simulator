"""The Problem — HOL Blocking: rendered from docs/problem.qmd"""
import os, re, streamlit as st
st.set_page_config(page_title="The Problem — HOL Blocking", page_icon="🔴", layout="wide")

def load_qmd(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"^---\n.*?\n---\n", "", content, count=1, flags=re.DOTALL)
    def replace_callout(m):
        icons = {"note":"📝","tip":"💡","warning":"⚠️","important":"❗",
                 "caution":"🔶","insight":"💡","danger":"🔴","success":"✅"}
        icon = icons.get(m.group(1), "📌")
        body = "\n> ".join(m.group(2).strip().split("\n"))
        return f"> {icon} **{m.group(1).title()}:**\n> {body}\n"
    content = re.sub(r"::: \{\.callout-(\w+)\}\n(.*?)\n:::", replace_callout, content, flags=re.DOTALL)
    content = re.sub(r"::: \{[^}]*\}\n", "", content)
    content = re.sub(r"\n:::\n", "\n", content)
    content = re.sub(r"^:::\s*$", "", content, flags=re.MULTILINE)
    return content.strip()

st.page_link("interactive_app.py", label="← Back to Simulator", icon="⚡")
qmd = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "problem.qmd")
st.markdown(load_qmd(qmd), unsafe_allow_html=True)

