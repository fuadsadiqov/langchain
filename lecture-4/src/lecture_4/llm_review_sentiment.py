from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

def main(review_text: str) -> None:
    class ReviewSetniment(BaseModel):
        sentiment: str = Field(
                description="The sentiment of review, either 'positive', 'negative', or 'neutral'."
        )

    promp_template = PromptTemplate(
        input_variables="review",
        template="""
            Analyze the sentiment of this product review:
            Classify it as 'positive', 'negative', or 'neutural'.

            Review text: 
            {review}
        """
    )
    llm_ollama = ChatOllama(model="deepseek-r1:1.5b", temperature=0.7).with_structured_output(ReviewSetniment)

    chain: ReviewSetniment = promp_template | llm_ollama

    response = chain.invoke({"review": review_text})
    print(response.sentiment)


main(
    "This product is normal. Not any extra advantages have and disadvantages"
)
