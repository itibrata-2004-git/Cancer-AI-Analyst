from app.agent import route_query


questions = [
    "What is cancer staging?",
    "What evidence is provided about cancer treatment?",
    "Compare cancer staging information across the documents.",
    "What is the weather in Delhi?"
]


for question in questions:
    result = route_query(question)

    print("\nQuestion:", question)
    print("Route:", result)
    