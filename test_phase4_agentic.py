from app.agent import run_agentic_workflow


questions = [
    "What is cancer staging?",
    "What evidence is provided about cancer treatment?",
    "Compare cancer staging information across the documents.",
    "What is the weather in Delhi?"
]


for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    result = run_agentic_workflow(question)

    print("ROUTE:", result["route"])
    print("ANSWER:")
    print(result["answer"][:500])