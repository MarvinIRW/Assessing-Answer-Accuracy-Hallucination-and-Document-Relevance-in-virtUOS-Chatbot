import yaml
import os

from src.config.core_config import Settings, settings


def set_config_language(language: str, config_file: str = "config.yaml") -> None:
    """
    Update the 'language' field in config.yaml and reset the 'settings' singleton
    so the next time creating an agent, it uses the new language.
    """
    # 1) Load the current config.yaml
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # 2) Update the 'language' key
    config_data["language"] = language
    print(f"Setting language to: {language}")

    # 3) Write updated config.yaml back to disk
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, allow_unicode=True)

    # 4) Force re-initialization of 'settings'
    #    This ensures the next call to CampusManagementOpenAIToolsAgent.run() re-reads config.yaml
    Settings._instance = None  # clear the singleton
    _ = Settings()  # re-init the config (so settings.language is updated)


import pandas as pd
import json
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from src.chatbot.agents.agent_openai_tools import CampusManagementOpenAIToolsAgent
from src.chatbot.tools.search_web_tool import visited_links

# Read the CSV
df = pd.read_csv("data/questions_dataset_reviewed_translated_de_en_cut.csv")
df_copy = df.copy()

all_results = []

for i, row in df_copy.iterrows():
    # ------------------------------------------------------------
    # 1) GET GERMAN QUESTION
    # ------------------------------------------------------------
    question_de = row["german_question_text_q"]
    print(question_de)
    question_id = row["question_id_q"]

    # Switch config.yaml -> language = "Deutsch"
    set_config_language("Deutsch", "config.yaml")  # path to your config.yaml
    # Now create a fresh agent with memory
    agent_executor_de = CampusManagementOpenAIToolsAgent.run(
        language="Deutsch",
        memory=ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=0
        ),
    )
    # Chatbot returns an answer in German
    result_de = agent_executor_de(question_de)
    visited_links_de = visited_links()

    # ------------------------------------------------------------
    # 2) GET ENGLISH QUESTION
    # ------------------------------------------------------------
    question_en = row["english_question_text_q"]
    print(question_en)

    # Switch config.yaml -> language = "English"
    set_config_language("English", "config.yaml")
    # Create a fresh agent with memory
    agent_executor_en = CampusManagementOpenAIToolsAgent.run(
        language="English",
        memory=ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=0
        ),
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
        "visited_links_de": visited_links_de,
        "question_en": question_en,
        "output_en": result_en["output"],
        "visited_links_en": visited_links_en,
    }
    all_results.append(row_dict)

# ------------------------------------------------------------
# 4) SAVE RESULTS
# ------------------------------------------------------------
with open("chatbot_responses.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

df_copy["chatbot_answer_de"] = [res["output_de"] for res in all_results]
df_copy["chatbot_answer_en"] = [res["output_en"] for res in all_results]
df_copy["chatbot_visited_urls_de"] = [
    ";".join(res["visited_links_de"]) for res in all_results
]
df_copy["chatbot_visited_urls_en"] = [
    ";".join(res["visited_links_en"]) for res in all_results
]

df_copy.to_csv("data/chatbot_answers_first_20_de_en.csv", index=False, quoting=1)
