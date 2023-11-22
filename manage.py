from transformers import AutoTokenizer, AutoModelForTokenClassification, BertForTokenClassification
import streamlit as st
import razdel
import torch
# Модель для поиска именованных сущностей от SpaCy из Hugging Face
import ru_core_news_lg


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
