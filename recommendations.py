from langchain.chains import LLMChain
from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate

API_KEY = 'AQVN3w-1YBjAAnkQV3p5tOSA1NA4thDkASturxJp'
ID = 'b1gjm0a8utlng73dcua8'

templates = {
    "en": (
        "Tell in detail how to care for a plant with the disease {disease}. "
        "Format the answer in Markdown. "
        "All text must be in English."
    ),
    "ru": (
        "Подробно расскажи, как ухаживать за растением с заболеванием {disease}. "
        "Ответ оформи в формате Markdown. "
        "Весь текст должен быть на русском языке."
    )
}

def ask(disease, lang="en"):
    template = templates[lang]
    prompt = PromptTemplate.from_template(template)
    llm = YandexGPT(api_key=API_KEY, folder_id=ID)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.invoke(disease)['text']

def get_recommendations(disease: str, lang="en"):
    if 'healthy' in disease.lower():
        return 'Your plant is healthy!' if lang == "en" else "Ваше растение здорово!"
    else:
        return ask(disease, lang)
