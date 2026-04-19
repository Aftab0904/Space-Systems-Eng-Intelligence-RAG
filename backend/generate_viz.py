import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def create_architecture_flowchart():
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define boxes
    boxes = [
        {"text": "NASA Systems Engineering\nHandbook (PDF)", "pos": (1, 8), "color": "#ff9999"},
        {"text": "LlamaIndex\nPyMuPDF Parser", "pos": (4, 8), "color": "#66b3ff"},
        {"text": "Hierarchical\nNode Parsing", "pos": (7, 8), "color": "#99ff99"},
        {"text": "Local BAAI\nEmbeddings (CPU)", "pos": (7, 5), "color": "#ffcc99"},
        {"text": "ChromaDB\nVector Store", "pos": (4, 5), "color": "#c2c2f0"},
        {"text": "User Natural\nLanguage Query", "pos": (1, 5), "color": "#ffb3e6"},
        {"text": "Groq Llama 3.3 70B\n(Reasoning Engine)", "pos": (1, 2), "color": "#c4e17f"},
        {"text": "Intelligent Technical\nResponse + Citations", "pos": (5.5, 2), "color": "#76D7C4"}
    ]

    # Draw boxes
    for box in boxes:
        rect = patches.FancyBboxPatch(
            (box["pos"][0]-0.8, box["pos"][1]-0.6), 1.6, 1.2, 
            boxstyle="round,pad=0.2", linewidth=2, edgecolor='black', facecolor=box["color"]
        )
        ax.add_patch(rect)
        ax.text(box["pos"][0], box["pos"][1], box["text"], ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows
    arrows = [
        ((1.8, 8), (3.2, 8)), # PDF -> Parser
        ((4.8, 8), (6.2, 8)), # Parser -> Hierarchical
        ((7, 7.4), (7, 5.6)), # Hierarchical -> Embeddings
        ((6.2, 5), (4.8, 5)), # Embeddings -> ChromaDB
        ((1.8, 5), (3.2, 5)), # User -> ChromaDB (Retrieval)
        ((1, 4.4), (1, 2.6)), # ChromaDB/Query -> Groq
        ((1.8, 2), (4.7, 2))  # Groq -> Response
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    plt.title("System Engineering Intelligence RAG: Architecture Flow", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("../assets/architecture_flow.png", dpi=300, bbox_inches='tight')
    print("Saved architecture_flow.png")

def create_tech_stack_boxes():
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ["Framework", "LLM", "Embeddings", "Database", "Parsing", "Frontend"]
    tools = ["LlamaIndex", "Groq (Llama 3.3)", "BAAI (Local)", "ChromaDB", "PyMuPDF", "Vanilla JS/CSS"]
    colors = sns.color_palette("husl", len(categories))

    y_pos = range(len(categories))
    bars = ax.barh(y_pos, [1]*len(categories), color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    
    for i, bar in enumerate(bars):
        ax.text(0.5, bar.get_y() + bar.get_height()/2, tools[i], 
                ha='center', va='center', color='black', fontsize=14, fontweight='bold')

    plt.title("Technology Stack Breakdown", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("../assets/tech_stack.png", dpi=300, bbox_inches='tight')
    print("Saved tech_stack.png")

if __name__ == "__main__":
    create_architecture_flowchart()
    create_tech_stack_boxes()
