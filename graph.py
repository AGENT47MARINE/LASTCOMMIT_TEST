from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    classifier_node, 
    summarizer_node, 
    entity_extractor_node, 
    rag_node, 
    code_solver_node,
    anomaly_detector_node,
    structured_processor_node,
    ambiguity_node,
    validator_node
)

# Initialize the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("classifier", classifier_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("entity_extractor", entity_extractor_node)
workflow.add_node("rag_engine", rag_node)
workflow.add_node("code_solver", code_solver_node)
workflow.add_node("anomaly_detector", anomaly_detector_node)
workflow.add_node("structured_processor", structured_processor_node)
workflow.add_node("ambiguity_handler", ambiguity_node)
workflow.add_node("validator", validator_node)

# Entry Point
workflow.set_entry_point("classifier")

# --- MULTI-TASK ROUTING LOGIC ---

def route_multi_task(state: AgentState):
    """
    Level 4: Parallel Router. 
    Returns a list of nodes to execute if multiple intents are found.
    """
    intents = state.get("intent", "").split(",")
    mapping = {
        "SUMMARIZE": "summarizer",
        "ENTITY": "entity_extractor",
        "RAG": "rag_engine",
        "CODE": "code_solver",
        "ANOMALY": "anomaly_detector",
        "STRUCTURED": "structured_processor"
    }
    
    # Identify which nodes to trigger
    target_nodes = [mapping[i.strip()] for i in intents if i.strip() in mapping]
    
    if not target_nodes:
        return "ambiguity_handler"
    
    return target_nodes

# Add Parallel Conditional Edges
workflow.add_conditional_edges(
    "classifier",
    route_multi_task,
    {
        "summarizer": "summarizer",
        "entity_extractor": "entity_extractor",
        "rag_engine": "rag_engine",
        "code_solver": "code_solver",
        "anomaly_detector": "anomaly_detector",
        "structured_processor": "structured_processor",
        "ambiguity_handler": "ambiguity_handler"
    }
)

# Connect all workers to the Validator (Fan-in)
# LangGraph automatically waits for all parallel branches to hit the same node
workflow.add_edge("summarizer", "validator")
workflow.add_edge("entity_extractor", "validator")
workflow.add_edge("rag_engine", "validator")
workflow.add_edge("code_solver", "validator")
workflow.add_edge("anomaly_detector", "validator")
workflow.add_edge("structured_processor", "validator")
workflow.add_edge("ambiguity_handler", "validator")

workflow.add_edge("validator", END)

# Compile
app = workflow.compile()
