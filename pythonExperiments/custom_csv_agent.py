from typing import Any, Dict, List, Optional
from langchain.llms import OpenAI

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.prompt import (
    PREFIX,
    SUFFIX_NO_DF,
    SUFFIX_WITH_DF,
)
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
# from langchain.load.serializable import PromptTemplate
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

import os
import logging

logging.basicConfig(level=logging.DEBUG)

# Path to your .env file
env_path = '../.env'

# Load the environment variables from the specified .env file
load_dotenv(dotenv_path=env_path)
load_dotenv()

def create_custom_tools_agent(
    llm: BaseLanguageModel,
    df: Any,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: Optional[bool] = True,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""
    try:
        import pandas as pd
    except ImportError:
        raise ValueError(
            "pandas package not found, please install with `pip install pandas`"
        )

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas object, got {type(df)}")
    if include_df_in_prompt is not None and suffix is not None:
        raise ValueError("If suffix is specified, include_df_in_prompt should not be.")
    if suffix is not None:
        suffix_to_use = suffix
        if input_variables is None:
            input_variables = ["df", "input", "agent_scratchpad"]
    else:
        if include_df_in_prompt:
            suffix_to_use = SUFFIX_WITH_DF
            input_variables = ["df", "input", "agent_scratchpad"]
        else:
            suffix_to_use = SUFFIX_NO_DF
            input_variables = ["input", "agent_scratchpad"]
    params = {
    "engine": "bing",
    "gl": "us",
    "hl": "en",
    }
    search = SerpAPIWrapper(params=params, serpapi_api_key=os.environ.get("SERPAPI_API_KEY"))
    tools = [PythonAstREPLTool(locals={"df": df}), Tool(name="search", description="search google", func=search.run)]
    
    logging.debug(f"============================prefix: {prefix}")
    logging.debug(f"============================suffix_to_use: {suffix_to_use}")
    logging.debug(f"============================input_variables: {input_variables}")
    logging.debug(f"============================tools: {tools}")
    logging.debug(f"============================kwargs being passed: {kwargs}")

    format_instructions = FORMAT_INSTRUCTIONS
    
    tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = format_instructions.format(tool_names=tool_names)
    template = "\n\n".join([prefix, tool_strings, format_instructions, suffix_to_use])
    if input_variables is None:
        input_variables = ["input", "agent_scratchpad"]

    # prompt = PromptTemplate(
    #     template=template, input_variables=input_variables
    # )

    # prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    # prompt.format(product="colorful socks")

    prompt = ZeroShotAgent.create_prompt(
        tools=tools, prefix=prefix, suffix=suffix_to_use, format_instructions=format_instructions, input_variables=input_variables
    )
    logging.debug("Prompt value: %s", prompt)

    if "df" in input_variables:
        partial_prompt = prompt.partial(df=str(df.head().to_markdown()))
    else:
        partial_prompt = prompt
    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        callback_manager=callback_manager,
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )

def create_custom_csv_agent(
    llm: BaseLanguageModel,
    path: str,
    pandas_kwargs: Optional[dict] = None,
    **kwargs: Any
) -> AgentExecutor:
    """Create csv agent by loading to a dataframe and using pandas agent."""
    import pandas as pd
    
    _kwargs = pandas_kwargs or {}
    df = pd.read_csv(path, **_kwargs)
    # df = pd.read_csv(path)
    return create_custom_tools_agent(llm, df, **kwargs)
    # return create_custom_tools_agent(llm, df)