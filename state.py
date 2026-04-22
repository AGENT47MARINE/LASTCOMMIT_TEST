from typing import TypedDict, Annotated, List, Optional
import operator

def merge_dicts(a: dict, b: dict) -> dict:
    """Reducer for merging results from parallel nodes."""
    res = (a or {}).copy()
    if b:
        res.update(b)
    return res

class AgentState(TypedDict):
    input: str
    intent: Optional[str]
    # Level 4 Update: result now merges instead of overwriting
    result: Annotated[Optional[dict], merge_dicts]
    confidence: float
    error: Optional[str]
    steps: Annotated[List[str], operator.add]
    reasoning: Annotated[List[str], operator.add]
    retries: int
