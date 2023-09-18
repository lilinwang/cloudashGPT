from custom_csv_agent import create_custom_csv_agent
from langchain.agents import create_csv_agent
#from langchain.llms import GPT4All
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Path to your .env file
env_path = '../.env'

# Load the environment variables from the specified .env file
load_dotenv(dotenv_path=env_path)
load_dotenv()

#model_path = os.environ.get("MODEL_PATH")
#llm = GPT4All(model=model_path, verbose=False)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, verbose=True)
agent = create_custom_csv_agent(llm=llm, path="pokemon.csv")
agent2 = create_csv_agent()
user_input = input()
print(agent.run(f"""
    If the query asks for a list of items, respond in this format where words are placed in quotes and numbers are not:
    "LIST ["item_1", "item_2", number_1, number_2, etc]"

    If the query asks for a line graph between two variables, respond in this format where x and y don't change:
    "LINE data={{[{{x: value_1, y: value_2}}, {{x: value_3, y: value_4}}, etc]}}"

    If the query asks for a scatter plot between two variables, respond in this format where x and y don't change and size is defaulted to 1,
    but if there is a third variable, change the size variable to that value:
    "SCATTER data={{[{{x: value_1, y: value_2, size: 1}}, {{x: value_3, y: value_4, size: 1}}, etc]}}"

    If the query asks to plot a bar graph, respond in this format where x and y don't change:
    "BAR1 data={{[{{x: "category_1", y: value_1}}, {{x: "category_2", y: value_2}}, etc]}}
    BAR2 data={{[{{x: "category_1", y: value_1}}, {{x: "category_2", y: value_2}}, etc]}}"
    If the query asks to plot a bar graph with multiple categories across two variables, repeat the above format for those categories.

    If the query asks to plot a pie chart, respond in this format:
    "PIE data={{[{{angle: value_1}}, {{angle: value_2}}, etc]}}"
    where each angle is its respective percent of 2 pi radians

    Otherwise, answer as needed.

    The following is the question: {user_input}
"""))
