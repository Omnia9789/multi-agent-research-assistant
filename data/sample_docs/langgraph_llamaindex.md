# Multi-Agent Systems and LangGraph

## What are Multi-Agent Systems?

Multi-agent systems (MAS) consist of multiple interacting intelligent agents. Each agent is an autonomous entity that perceives its environment and takes actions to achieve its goals. Agents may cooperate, compete, or operate independently.

In the context of LLM-powered systems, agents are specialized LLM instances or tools that handle specific subtasks.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain and provides:

- **State Management**: A typed state object flows through the graph and is updated by each node.
- **Node Definitions**: Each node is a Python function (sync or async) that takes state and returns updated state.
- **Edge Routing**: Edges can be static or conditional (routing based on state values).
- **Cycles**: Unlike DAGs, LangGraph supports cycles, enabling retry loops and iterative refinement.

### Key Concepts

**StateGraph**: The primary graph class. Define nodes and edges, then compile to get a runnable.

**Conditional Edges**: Route to different nodes based on state. Example:
```python
graph.add_conditional_edges("fact_check", router_fn, {"pass": "output", "fail": "retry"})
```

**Checkpointing**: LangGraph supports persistence via checkpointers (SQLite, Redis), enabling resumable workflows.

## Agent Orchestration Patterns

### Sequential Pipeline
Agents run in a fixed order: A → B → C → D. Simple and predictable.

### Fan-Out / Fan-In
A coordinator agent dispatches to multiple specialist agents in parallel, then merges results.

### Iterative Refinement
An evaluator agent loops back to a generator agent until quality criteria are met.

### Hierarchical Agents
A supervisor agent dynamically selects which sub-agent to invoke based on the task.

## LlamaIndex

LlamaIndex (formerly GPT Index) is a data framework for LLM applications:
- **Document Loading**: Supports 100+ data connectors (PDFs, databases, APIs, etc.)
- **Indexing**: Vector stores, keyword indexes, knowledge graphs.
- **Querying**: High-level query engines with customizable retrieval and synthesis.
- **Agents**: Tool-using agents built on top of LlamaIndex components.

### VectorStoreIndex
The most common index type. Documents are chunked, embedded, and stored in a vector database. At query time, the top-k most similar chunks are retrieved.

### Integration with LangGraph
LlamaIndex handles the RAG heavy lifting (indexing + retrieval) while LangGraph orchestrates the overall multi-step workflow. This combination provides both robust retrieval and flexible agent coordination.
