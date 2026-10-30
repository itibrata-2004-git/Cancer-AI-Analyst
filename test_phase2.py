
import app.agent as m

print("\n" + "=" * 55)
print("CANCER AI ANALYST - PHASE 2 ADAPTIVE RAG TEST")
print("=" * 55)

question = (
    "What does the uploaded medical literature say "
    "about the definition of cancer?"
)

print("\nQuestion:")
print(question)

print("\nRunning Adaptive RAG workflow...")

result = m.run_adaptive_rag(question)

print("\nRetrieval attempts:")
print(result["attempts"])

print("\nEvidence sufficient:")
print(result["sufficient"])

print("\nEvidence assessment:")
print(result["assessment"])

print("\nRetrieved sources:")

for document in result["evidence"]:
    metadata = document.metadata or {}

    filename = metadata.get(
        "filename",
        metadata.get("source", "Unknown")
    )

    page = metadata.get("page")

    if page is not None:
        try:
            page = int(page) + 1
        except (TypeError, ValueError):
            pass

    print(f"- {filename}, page {page}")

print("\nFinal answer:")
print(result["answer"])

print("\nPhase 2 test completed.")
