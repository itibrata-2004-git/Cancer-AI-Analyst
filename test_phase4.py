from app.agent import classify_query


questions = [
    "What is cancer staging?",
    "What evidence is provided about cancer treatment?",
    "Compare the cancer staging information across the documents.",
    "What is the weather in Delhi?"
]


for question in questions:
    result = classify_query(question)

    print("\nQuestion:", question)
    print("Classification:", result)