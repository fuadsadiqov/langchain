from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

def main() -> None: 
    class ProductOutput(BaseModel):
        headline: str = Field(
            description="Catchy headline"
        )
        call_to_action: str = Field(
            description="Call to action"
        )
        body: str = Field(
            description="Marketing copy body"
        )

    class TranslatedCopy(BaseModel):
        translated_headline: str = Field(description="Translated headline")
        translated_body: str = Field(description="Translated body")
        translated_cta: str = Field(description="Translated call to action")

    copy_prompt = PromptTemplate(
        input_variables="product",
        template="""
            Create compelling marketing copy for. {product}

            Include: 
            - Attention-grabbing headline
            - Persuasive body text (2-3 sentences)
            - Strong call to action
        """
    )
    llm_ollama = ChatOllama(model="deepseek-r1:1.5b", temperature=0.7)

    copy_chain = copy_prompt | llm_ollama.with_structured_output(ProductOutput)

    marketing_copy = copy_chain.invoke({"product": "Mercedes Benz E300"})

    def validate_marketing_copy_gate(copy: ProductOutput) -> ProductOutput:        
        issues = []

        print(copy)
        if len(copy.headline.split()) < 3:
            issues.append("Headline too short")
            
        if len(copy.body.split()) < 10:
            issues.append("Body text too short")
        
        if issues:
            print(f"❌ Gate FAILED: {'; '.join(issues)}")
            raise ValueError(f"Marketing copy validation failed: {issues}")
            
        print("✅ Gate PASSED: Marketing copy approved")
        return copy

    validated_marketing_copy = validate_marketing_copy_gate(marketing_copy)

    translation_prompt = PromptTemplate(
        input_variables=["headline", "body", "cta", "target_language"],
        template="""
            Translate this contents to {target_language}.

            Headline: {headline},
            Body: {body},
            Call to action: {cta}
        """
    )
    translation_llm_ollama = ChatOllama(model="deepseek-r1:1.5b", temperature=0)

    raw = translation_llm_ollama.invoke(
        translation_prompt.format(
            headline=validated_marketing_copy.headline,
            body=validated_marketing_copy.body,
            cta=validated_marketing_copy.call_to_action,
            target_language="Turkish"
        )
    )
    print(raw.content)

main()