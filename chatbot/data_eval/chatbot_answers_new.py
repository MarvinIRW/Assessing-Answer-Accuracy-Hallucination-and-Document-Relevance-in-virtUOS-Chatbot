import sys

# print(f"Python version: {sys.path}")
sys.path.append("./")

import pandas as pd
import json
from src.chatbot.agents.agent_openai_tools import CampusManagementOpenAIToolsAgent
from src.chatbot.tools.search_web_tool import visited_links
from src.config.core_config import settings
from typing import List
from datetime import datetime
# Ignore the warning in on_llm_new_token
import logging
class SuppressSpecificWarning(logging.Filter):
    def filter(self, record):
        return "unsupported operand type(s) for -" not in record.getMessage()


logger = logging.getLogger("langchain_core.callbacks.manager")
logger.addFilter(SuppressSpecificWarning())
####################################
# Exceptions for the agent
class AgentException(Exception):
    """Custom exception for agent-related issues."""
    pass

# Function to compare memory states
def assert_memory_states(agent_executor, initial_memory):
    """
    Checks if the agent's memory after .clear() matches the initial memory state.
    """
    if agent_executor.memory != initial_memory:
        raise AgentException(
            f"Memory mismatch for agent {id(agent_executor)}: "
            f"Expected {initial_memory}, but got {agent_executor.memory}."
        )
    
# Function to check if visited links are an empty list
def assert_visited_links_empty():
    """
    Checks if the visited links list is empty.
    """
    if visited_links():
        raise AgentException(
            "Visited links list is not empty after clearing."
        )
    
# check if the language is set to English or German
def check_language(agent_executor, language):
    """
    Checks if the agent's language is set to the correct language.
    """
    if agent_executor.language != language:
        raise AgentException(
            f"Language mismatch for agent {id(agent_executor)}: "
            f"Expected {language}, but got {agent_executor.language}."
        )
# Function to check for overlaps between queries and DataFrame questions
def assert_no_overlap_with_df(queries: List[str], df: pd.DataFrame, question_column: str):
    """
    Ensures that no query in the provided list is present in the specified column of the DataFrame.

    Parameters:
    queries (List[str]): The list of queries to check.
    df (pd.DataFrame): The DataFrame containing questions.
    question_column (str): The column name in the DataFrame to check against.

    Raises:
    AgentException: If any overlap is found.
    """
    overlapping_queries = set(queries).intersection(df[question_column].dropna())
    if overlapping_queries:
        raise AgentException(
            f"Overlapping queries detected in {question_column}: {overlapping_queries}"
        )

####################################

# Create a fresh German agent
settings.language = "Deutsch"
agent_executor_de = CampusManagementOpenAIToolsAgent.run(language="Deutsch")
initial_memory_de = agent_executor_de.memory.copy()  # Save the initial memory state
print(visited_links())  # Save the initial visited links

# Create a fresh English agent
settings.language = "English"
agent_executor_en = CampusManagementOpenAIToolsAgent.run(language="English")
initial_memory_en = agent_executor_en.memory.copy()  # Save the initial memory state

# Check if the two agents are the same
if id(agent_executor_de) == id(agent_executor_en):
    raise AgentException("The two agents share the same memory address, which is not expected!")

warm_up_queries_en = [
    "Hello! How can I get started with my application?",
    "How do I reset my university account password?",
    "How do I apply for a scholarship?",
    "How do I apply for a student visa?",
    "How do I apply for a student loan?",
    "How do I apply for a student housing?",
]

warm_up_queries_de =[
    "Hallo! Wie kann ich mit meiner Bewerbung beginnen?",
    "Wie setze ich mein Universitätskontopasswort zurück?",
    "Wie bewerbe ich mich um ein Stipendium?",
    "Wie bewerbe ich mich um ein Studentenvisum?",
    "Wie bewerbe ich mich um ein Studentendarlehen?",
    "Wie bewerbe ich mich um eine Studentenwohnung?",
]


def warm_up_agent(warm_up_queries: List[str], agent_executor) -> None:
    """
    Downloads models and  prepare the agent for actual tasks.

    Parameters:
    warm_up_queries (List[str]): A list of queries to be executed during the warm-up phase.

    Returns:
    None
    """
    # Warm-up phase
    for q in warm_up_queries:
        agent_executor(q)

# Warm-up phase
warm_up_agent(warm_up_queries_de, agent_executor_de)
warm_up_agent(warm_up_queries_en, agent_executor_en)


# Clear memory and verify states
print("Clearing memory and links...")
agent_executor_de.memory.clear()
agent_executor_en.memory.clear()
visited_links.clear()

print("Verifying memory states and links...")
assert_memory_states(agent_executor_de, initial_memory_de)
assert_memory_states(agent_executor_en, initial_memory_en)
assert_visited_links_empty()

print("All checks passed successfully.")

# Read the CSV
df = pd.read_csv("data_eval/chatbot_questions.csv")
df_copy = df.copy()
#df_copy = df.head(1).copy()  # for testing

all_results = []

# Check for overlaps in German and English queries
# Could interfere with the context retrieval
assert_no_overlap_with_df(warm_up_queries_de, df_copy, "german_question_text_q")
assert_no_overlap_with_df(warm_up_queries_en, df_copy, "english_question_text_q")

# save the start time for context retrieval
start_time = datetime.now()

for i, row in df_copy.iterrows():
    # ------------------------------------------------------------
    # 1) GET GERMAN QUESTION
    # ------------------------------------------------------------
    question_de = row["german_question_text_q"]
    question_id = row["question_id_q"]
    
    # Clear the memory
    agent_executor_de.memory.clear()
    assert_memory_states(agent_executor_de, initial_memory_de)
    # Chatbot returns an answer in German
    check_language(agent_executor_de, "Deutsch")
    result_de = agent_executor_de(question_de)
    visited_links_de = visited_links()
    # clear visited links
    visited_links.clear()
    assert_visited_links_empty()

    # ------------------------------------------------------------
    # 2) GET ENGLISH QUESTION
    # ------------------------------------------------------------
    question_en = row["english_question_text_q"]

    # Clear the memory
    agent_executor_en.memory.clear()
    assert_memory_states(agent_executor_en, initial_memory_en)
    # Chatbot should now return an answer in English
    check_language(agent_executor_en, "English")
    result_en = agent_executor_en(question_en)
    visited_links_en = visited_links()
    # clear visited links
    visited_links.clear()
    assert_visited_links_empty()

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
        "earliest_start_time": start_time,
    }
    all_results.append(row_dict)
    
# ------------------------------------------------------------
# 4) SAVE RESULTS
# ------------------------------------------------------------
# with open("data_eval/chatbot_responses_test.json", "w", encoding="utf-8") as f:
#     json.dump(all_results, f, indent=2, ensure_ascii=False)

df_copy["chatbot_answer_de"] = [res["output_de"] for res in all_results]
df_copy["chatbot_answer_en"] = [res["output_en"] for res in all_results]
df_copy["chatbot_links_function_de"] = [
    ";".join(res["visited_links_function_de"]) for res in all_results
]
df_copy["chatbot_links_function_en"] = [
    ";".join(res["visited_links_function_en"]) for res in all_results
]
df_copy["earliest_start_time"] = start_time

df_copy.to_csv(
    "data_eval/chatbot_answers.csv", index=False, quoting=1
)
