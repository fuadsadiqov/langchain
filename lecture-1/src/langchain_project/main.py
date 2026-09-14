from langchain_core.prompts import PromptTemplate
from langchain_anthropic.chat_models import ChatAnthropic

def main() -> None:
    prompt_template = PromptTemplate(
        input_value=["name"],
        template="Sabahın xeyir {name}, necəsən?"
    )

    llm_anthropic = ChatAnthropic(
        model_name='claude-haiku-4-5-20251001',
        temperature=0.7
    )

    chain = prompt_template | llm_anthropic 

    response = chain.invoke({"name": "Fuad"})
    print(response.content)

main()