import app.agent as m


print("\n" + "=" * 60)
print("CANCER AI ANALYST - PHASE 3 RESEARCH ASSISTANT TEST")
print("=" * 60)


question = (
    "Compare the information about cancer staging "
    "available in the uploaded medical documents."
)


print("\nResearch question:")
print(question)

print("\nRunning Research Assistant...")

result = m.run_research_assistant(question)


print("\n" + "=" * 60)
print("RESEARCH SUMMARY")
print("=" * 60)

print(result["summary"])


print("\n" + "=" * 60)
print("RETRIEVED SOURCES")
print("=" * 60)

for source in result["sources"]:

    filename = source["filename"]
    page = source["page"]

    print(
        f"- {filename}, page {page}"
    )


print("\nPhase 3 test completed.")