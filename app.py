import streamlit as st
from PIL import Image
from model import load_models, predict
from recommendations import get_recommendations
from preprocessing import preprocess_images

N = 3  # кол-во изображений для загрузки

translations = {
    "en": {
        "title": "Disease Detection",
        "upload": f"Upload {N} images of the plant",
        "preprocess": "Preprocessing images",
        "preprocess_done": "Preprocessing done",
        "predicting": "Generating a response",
        "predicted": "Response generated",
        "recom": "Generating recommendations",
        "recom_done": "Recommendations generated",
        "disease": "Disease Diagnosis:",
        "recommend": "Recommendations:",
        "warning": f"Please upload exactly {N} images.",
        "language": "Select interface language"
    },
    "ru": {
        "title": "Детектор заболеваний",
        "upload": f"Загрузите {N} изображений растения",
        "preprocess": "Обработка изображений",
        "preprocess_done": "Обработка завершена",
        "predicting": "Генерация ответа",
        "predicted": "Ответ сгенерирован",
        "recom": "Формирование рекомендаций",
        "recom_done": "Рекомендации сформированы",
        "disease": "Диагноз заболевания:",
        "recommend": "Рекомендации:",
        "warning": f"Пожалуйста, загрузите ровно {N} изображений.",
        "language": "Выберите язык интерфейса"
    }
}

@st.cache_resource
def get_models():
    return load_models()

models = get_models()

def main():
    lang = st.selectbox("🌐", ["English", "Русский"], index=0)
    lang_code = "en" if lang == "English" else "ru"
    t = translations[lang_code]  

    st.title(t["title"])

    uploaded_files = st.file_uploader(
        t["upload"], type=["jpg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) == N:
            raw_images = [Image.open(file) for file in uploaded_files]

            col = st.columns(N)
            for img in range(len(raw_images)):
                with col[img]:
                    st.image(raw_images[img], use_container_width=True)

            with st.spinner(t["preprocess"]):
                images = preprocess_images(raw_images)
                st.success(t["preprocess_done"])

            with st.spinner(t["predicting"]):
                result = predict(images, models)
                st.success(t["predicted"])

            with st.spinner(t["recom"]):
                recom = get_recommendations(result, lang_code)
                st.success(t["recom_done"])

            st.subheader(t["disease"])
            st.write(result)

            st.subheader(t["recommend"])
            st.write(recom)
        else:
            st.warning(t["warning"])

if __name__ == "__main__":
    main()
