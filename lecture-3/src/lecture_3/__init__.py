from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import requests
from langchain_core.tools import tool

def run_with_no_tool(prompt: str) -> str:
    prompt_tepmlate = PromptTemplate.from_template(
        "Sen bir yapay zekasın. Soruyu cevapla: {question}"
    )
    llm = ChatOllama(model="deepseek-r1:1.5b",temperature=0)
    output_parser = StrOutputParser()

    chain = (
        prompt_tepmlate |
        llm | 
        output_parser
    )
    return chain.invoke({"question": prompt})

def run_with_tools(prompt: str) -> str:
    prompt_tepmlate = PromptTemplate.from_template(
        "Sen bir yapay zekasın. Soruyu cevapla: {question}"
    )
    llm = ChatOllama(model="deepseek-r1:1.5b",temperature=0)
    llm_with_tools = llm.bind_tools([get_current_weather])

    chain = (
        prompt_tepmlate |
        llm_with_tools 
    )
    response = chain.invoke({"question": prompt})
    print(f"\n{response}")
    return response

@tool
def get_current_weather() -> dict:
    """
    Get the current weather for the user's location.
    """
    ip_response = requests.get("https://ipinfo.io/json")
    ip_data = ip_response.json()

    latitude, longitude = ip_data["loc"].split(",")

    weather_response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    )

    weather_data = weather_response.json()
    return weather_data["current_weather"]  


def main() -> None:
    prompt: str= "Şu anki hava kaç derece?"

    # response_llm_with_no_tools = run_with_no_tool(prompt)

    response_llm_with_tools = run_with_tools(prompt)

    # print("response_llm_with_no_tools",  response_llm_with_no_tools)
    print("response_llm_with_tools", response_llm_with_tools)


main()