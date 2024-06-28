from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load pre-trained model and tokenizer from the local directory
tokenizer = BertTokenizer.from_pretrained('model')
model = BertForQuestionAnswering.from_pretrained('model')

def get_response(question):
    # Tokenize input
    inputs = tokenizer(question, return_tensors='pt')

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Process and return response
    answer = tokenizer.decode(outputs[0].argmax(dim=-1))
    return answer
