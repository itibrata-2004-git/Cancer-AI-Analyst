from app.agent import (
    run_agentic_workflow,
    evaluate_agentic_result,
)


questions = [
    "What is cancer staging?",
    "What evidence is provided about cancer treatment?",
    "Compare cancer staging information across the documents.",
    "What is the weather in Delhi?",
]


for question in questions:

    print("\n" + "=" * 70)

    print("QUESTION:")
    print(question)

    result = run_agentic_workflow(
        question
    )

    evaluation = evaluate_agentic_result(
        question,
        result
    )

    print("\nROUTE:")
    print(evaluation["route"])

    print("\nANSWER GENERATED:")
    print(evaluation["answer_generated"])

    print("\nANSWER LENGTH:")
    print(evaluation["answer_length"])

    print("\nEVIDENCE COUNT:")
    print(evaluation["evidence_count"])

    print("\nSOURCE COUNT:")
    print(evaluation["source_count"])

    print("\nEVIDENCE SUPPORTED:")
    print(evaluation["evidence_supported"])