

import streamlit as st
# Модель для поиска именованных сущностей от SpaCy из Hugging Face
import ru_core_news_md



nlp = ru_core_news_md.load()
def searchNER(text):
    doc = nlp(text)
    colors = {"PER": "#9966cc", "ORG": "green", "LOC": "#1dacd6", "Date": "red", "Money": "yellow"}
    edited_text = ""
    for token in doc:
        if token.ent_type_ in colors.keys():
            edited_text += "<span style=\"color:" + colors[token.ent_type_] + ";\">"+ token.text +"</span> "
        else:
            edited_text += token.text + ' '
    return edited_text



st.title("Приложение для поиска именнованных сущностей (NER)")
input_text = st.text_area("Введите текст")

st.write('Цветовое обозночение NER:')
st.write('| Имена 🟣 |', ' Организации 🟢 |', ' Локации 🔵 |', ' Валюта 🟡 |', 'Даты 🔴 |')

if st.button("Начать"):
    with st.spinner('Обработка...'):
        result = searchNER(input_text)
        st.header('Текст с выделенными NER')
        st.markdown(result, unsafe_allow_html=True)
