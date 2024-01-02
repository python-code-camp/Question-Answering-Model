from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a question-answering pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

context = "Sachin Tendulkar, often referred to as the 'God of Cricket'," \
          " is a legendary figure in the world of sports. Born on April 24, " \
          "1973, in Mumbai, India, Tendulkar's name is synonymous with excellence " \
          "and mastery in the game of cricket. His remarkable career spanned 24 years, " \
          "during which he set numerous records and achieved unparalleled success."

while True:
    user_input = input("Ask a question (or press 'q' to quit): ")

    if user_input.lower() == 'q':
        break

    QA_input = {
        'question': user_input,
        'context': context
    }

    res = nlp(QA_input)
    print(res)

    print(f"Answer: {res['answer']}")
