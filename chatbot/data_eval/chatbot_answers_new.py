import sys

# print(f"Python version: {sys.path}")
sys.path.append("./")

import pandas as pd
import json
from src.chatbot.agents.agent_openai_tools import CampusManagementOpenAIToolsAgent
from src.chatbot.tools.search_web_tool import visited_links
from src.config.core_config import settings

# Read the CSV
df = pd.read_csv("data_eval/chatbot_questions.csv")
# df_copy = df.head(1).copy() for testing
df_copy = df.copy()

all_results = []

for i, row in df_copy.iterrows():
    # ------------------------------------------------------------
    # 1) GET GERMAN QUESTION
    # ------------------------------------------------------------
    question_de = row["german_question_text_q"]
    question_id = row["question_id_q"]
    settings.language = "Deutsch"  # This will also reset the history

    # Now create a fresh agent with memory
    agent_executor_de = CampusManagementOpenAIToolsAgent.run(language="Deutsch")
    # Chatbot returns an answer in German
    result_de = agent_executor_de(question_de)
    visited_links_de = visited_links()

    # ------------------------------------------------------------
    # 2) GET ENGLISH QUESTION
    # ------------------------------------------------------------
    question_en = row["english_question_text_q"]
    settings.language = "English"  # This will also reset the history

    # Create a fresh agent with memory
    agent_executor_en = CampusManagementOpenAIToolsAgent.run(
        language="English",
    )
    # Chatbot should now return an answer in English
    result_en = agent_executor_en(question_en)
    visited_links_en = visited_links()

    # ------------------------------------------------------------
    # 3) Combine results
    # ------------------------------------------------------------
    row_dict = {
        "question_id": question_id,
        "question_de": question_de,
        "output_de": result_de["output"],
        "visited_links_function_de": visited_links_de,
        "question_en": question_en,
        "output_en": result_en["output"],
        "visited_links_function_en": visited_links_en,
    }
    all_results.append(row_dict)

# ------------------------------------------------------------
# 4) SAVE RESULTS
# ------------------------------------------------------------
with open("data_eval/chatbot_responses.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

df_copy["chatbot_answer_de"] = [res["output_de"] for res in all_results]
df_copy["chatbot_answer_en"] = [res["output_en"] for res in all_results]
df_copy["chatbot_links_function_de"] = [
    ";".join(res["visited_links_function_de"]) for res in all_results
]
df_copy["chatbot_links_function_en"] = [
    ";".join(res["visited_links_function_en"]) for res in all_results
]

df_copy.to_csv(
    "data_eval/chatbot_answers.csv", index=False, quoting=1
)  # 08.01.2025 19:22
