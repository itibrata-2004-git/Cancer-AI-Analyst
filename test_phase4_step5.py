from app.agent import run_research_assistant


question = "Compare cancer staging information across the documents."

result = run_research_assistant(question)

print("\n" + "=" * 70)
print("QUESTION:")
print(question)

print("\nEVIDENCE VALIDATION:")
print(result.get("evidence_validation"))

print("\nRESEARCH ANSWER:")
print(result.get("summary", ""))