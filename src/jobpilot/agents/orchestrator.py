"""LangGraph StateGraph: evaluate → (cond on score) → tailor → END."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jobpilot.config import Settings
from jobpilot.logging_setup import get_logger
from jobpilot.models.state import AgentState

if TYPE_CHECKING:
    from jobpilot.agents.evaluator import EvaluatorAgent
    from jobpilot.agents.tailor import TailorAgent

log = get_logger(__name__)


def build_graph(
    *,
    settings: Settings,
    evaluator: EvaluatorAgent,
    tailor: TailorAgent,
) -> CompiledStateGraph[AgentState]:
    """Compile the two-node JobPilot graph."""

    async def evaluate_node(state: AgentState) -> AgentState:
        return await evaluator.run(state)

    async def tailor_node(state: AgentState) -> AgentState:
        return await tailor.run(state)

    def decide(state: AgentState) -> str:
        evaluation = state.get("evaluation")
        if evaluation is None:
            log.warning("orchestrator.no_evaluation_result")
            return "skip"
        if evaluation.score >= settings.score_threshold:
            log.info("orchestrator.tailor", score=evaluation.score)
            return "tailor"
        log.info("orchestrator.skip", score=evaluation.score)
        return "skip"

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("tailor", tailor_node)
    graph.add_edge(START, "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        decide,
        {"tailor": "tailor", "skip": END},
    )
    graph.add_edge("tailor", END)
    return graph.compile()
