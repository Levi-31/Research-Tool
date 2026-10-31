from langgraph.graph import StateGraph, START, END
from blog_generator.models import State

# Import node functions and subgraphs
from blog_generator.nodes.router import router_node, route_next
from blog_generator.nodes.research import research_node
from blog_generator.nodes.orchestrator import orchestrator_node, fanout
from blog_generator.nodes.worker import worker_node
from blog_generator.nodes.reducer import reducer_subgraph

# Build the main graph
g = StateGraph(State)

# Add nodes
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

# Add edges and conditional routing logic
g.add_edge(START, "router")
g.add_conditional_edges(
    "router", 
    route_next, 
    {"research": "research", "orchestrator": "orchestrator"}
)
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

# Compile the final application
app = g.compile()
