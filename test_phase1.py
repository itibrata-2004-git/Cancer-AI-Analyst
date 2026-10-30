
import app.agent as m


print("\n" + "=" * 55)
print("CANCER AI ANALYST - PHASE 1 TOOL TEST")
print("=" * 55)

# --------------------------------------------------
# TEST 1: Check registered tools
# --------------------------------------------------

print("\nTEST 1: Registered tools")

for tool in m.agent_tools:
    print("-", tool.name)

# --------------------------------------------------
# TEST 2: Search medical documents
# --------------------------------------------------

print("\nTEST 2: FAISS medical document search")

search_result = m.search_medical_documents.invoke({
    "query": "cancer research"
})

print(search_result[:2000])

# --------------------------------------------------
# TEST 3: Look up an indexed document
# --------------------------------------------------

print("\nTEST 3: Document lookup")

stored_documents = getattr(
    m.vectorstore.docstore,
    "_dict",
    {}
)

if stored_documents:
    first_document = next(
        iter(stored_documents.values())
    )

    metadata = first_document.metadata or {}

    filename = metadata.get("filename")

    if not filename:
        import os
        filename = os.path.basename(
            str(metadata.get("source", ""))
        )

    if filename:
        lookup_result = m.lookup_document.invoke({
            "filename": filename
        })

        print(lookup_result[:1500])
    else:
        print("No filename found in document metadata.")
else:
    print("FAISS index contains no documents.")

# --------------------------------------------------
# TEST 4: Evidence checker
# --------------------------------------------------

print("\nTEST 4: Evidence verification")

evidence_result = m.check_evidence.invoke({
    "claim": "The document explicitly states that the study "
             "included 100 participants.",
    "evidence": "The study included 100 participants."
})

print(evidence_result)

# --------------------------------------------------
# TEST 5: Full LangGraph agent
# --------------------------------------------------

print("\nTEST 5: LangGraph agent")

question = (
    "Use lookup_document to inspect Cancer_basics.pdf. "
    "Then use search_medical_documents to retrieve relevant "
    "evidence about the definition of cancer. "
    "Use check_evidence to evaluate whether the retrieved "
    "evidence supports the statement that cancer involves "
    "uncontrolled cell growth. "
    "Summarize the findings and cite the source pages."
)

result = m.agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
})
print("\nTOOLS ACTUALLY CALLED:")

for message in result["messages"]:
    tool_calls = getattr(message, "tool_calls", [])

    for tool_call in tool_calls:
        print(
            "Tool:",
            tool_call.get("name"),
            "| Arguments:",
            tool_call.get("args")
        )
print("\nAgent response:")

for message in reversed(result["messages"]):
    if message.type == "ai" and message.content:
        print(message.content)
        break

print("\nPhase 1 testing completed.")
