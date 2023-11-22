from transformers import AutoTokenizer, AutoModelForTokenClassification, BertForTokenClassification
import streamlit as st
import razdel
import torch
# Модель для поиска именованных сущностей от SpaCy из Hugging Face
import ru_core_news_md


# суммаризация (реферирование)
summarization_model_name = "IlyaGusev/rubert_ext_sum_gazeta"
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
sep_token = summarization_tokenizer.sep_token
sep_token_id = summarization_tokenizer.sep_token_id

sum_model = BertForTokenClassification.from_pretrained(summarization_model_name)

if torch.cuda.is_available():
    device = torch.device("cuda", 1)
    print('GPU avaliable')
else:
    device = torch.device("cpu")
    print("GPU Unavaliable")

sum_model.to(device="cpu")

def summarize(text, ratio=50):
    sentences = [s.text for s in razdel.sentenize(text)]
    sentences_count = len(sentences)
    text = sep_token.join(sentences)
    inputs = summarization_tokenizer(
        [text], max_length=500, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    sep_mask = inputs["input_ids"][0] == sep_token_id
    current_token_type_id = 0 
    for pos, input_id in enumerate(inputs["input_ids"][0]):
        inputs["token_type_ids"][0][pos] = current_token_type_id
        if input_id == sep_token_id:
            current_token_type_id = 1 - current_token_type_id      
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    with torch.no_grad(): 
        outputs = sum_model(**inputs) 
    logits = outputs.logits[0, :, 1]

    logits = logits[sep_mask]
    logits, indices = logits.sort(descending=True)
    logits, indices = logits.cpu().tolist(), indices.cpu().tolist()
    pairs = list(zip(logits, indices))
    length = int(sentences_count*(ratio/100))
    pairs = pairs[:length]
    indices = list(sorted([idx for _, idx in pairs]))
    summary = " ".join([sentences[idx] for idx in indices])
    return summary


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


st.title("Приложение для реферирования текста и поиска именнованных сущностей (NER)")

input_text = st.text_area("Введите текст:")
ratio = st.slider('Выберите процент сокращения текста', 0, 100, 50)
input_text_ner = st.checkbox('Поиск NER в исходном тексте')
output_text_ner = st.checkbox('Поиск NER в сокращенном тексте')

st.write('Цветовое обозночение NER:')
st.write('| Имена 🟣 |', ' Организации 🟢 |', ' Локации 🔵 |', ' Валюта 🟡 |', 'Даты 🔴 |')

if st.button("Реферировать"):
    with st.spinner('Обработка...'):
        summary = summarize(input_text, ratio)
        if input_text_ner:
            result = searchNER(input_text)
            st.header('Исходный текст с выделенными NER')
            st.markdown(result, unsafe_allow_html=True)

        if output_text_ner:
            result = searchNER(summary)
            st.header('Сокращенный текст с выделенными NER')
            st.markdown(result, unsafe_allow_html=True)
        else:
            st.header('Сокращенный текст')
            st.write(summary)