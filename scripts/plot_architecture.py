import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

# Create directed graph
G = nx.DiGraph()

# Nodes (steps/components)
G.add_node("Dataset (JSONL from Clang Tidy/Tree Sitter)", shape='box')
G.add_node("Preprocessing (Prompt-Completion Pairs + System Prompt)", shape='box')
G.add_node("QLoRA/PEFT Fine-Tuning", shape='ellipse')
G.add_node("Variant 1: Detection Model (Smell JSON Output)", shape='box')
G.add_node("Variant 2: Fixing Model (Fixed Code + Explanations JSON)", shape='box')
G.add_node("Inference Pipeline (GPU)", shape='ellipse')
G.add_node("Structured JSON Outputs", shape='box')

# Edges (flow)
G.add_edge("Dataset (JSONL from Clang Tidy/Tree Sitter)", "Preprocessing (Prompt-Completion Pairs + System Prompt)")
G.add_edge("Preprocessing (Prompt-Completion Pairs + System Prompt)", "QLoRA/PEFT Fine-Tuning")
G.add_edge("QLoRA/PEFT Fine-Tuning", "Variant 1: Detection Model (Smell JSON Output)")
G.add_edge("QLoRA/PEFT Fine-Tuning", "Variant 2: Fixing Model (Fixed Code + Explanations JSON)")
G.add_edge("Variant 1: Detection Model (Smell JSON Output)", "Inference Pipeline (GPU)")
G.add_edge("Variant 2: Fixing Model (Fixed Code + Explanations JSON)", "Inference Pipeline (GPU)")
G.add_edge("Inference Pipeline (GPU)", "Structured JSON Outputs")

# Draw with layout
pos = graphviz_layout(G, prog='dot')  # Hierarchical layout
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', arrows=True)
plt.title("Overall System Architecture: Fine-Tuning + Inference")
plt.savefig("system_architecture.png")  # Save as PNG
plt.show()  # Display inline