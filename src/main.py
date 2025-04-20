# This is a simple example of calling an LLM with LangChain.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# This expects a .env file in the same directory, containing
"""
OPENAI_API_KEY="<your api key here>"
"""


def basic_chain():
    prompt = ChatPromptTemplate.from_template(
        "Tell me the most noteworthy books by the author {author}"
    )
    model = ChatOpenAI(model="gpt-4")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    results = chain.invoke({"author": "William Faulkner"})
    print(results)


def main():
    load_dotenv()
    basic_chain()


if __name__ == "__main__":
    main()
