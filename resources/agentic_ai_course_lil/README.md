# Agentic AI Course - Action & Planning Autonomy

Interactive Jupyter notebooks teaching agentic AI concepts from first principles.

## 📚 Notebooks

### V1: Action Autonomy - Router Agent
**File:** `v1_action_autonomy.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agentic_ai_course_lil/v1_action_autonomy.ipynb)

Learn to build an AI agent that performs single, well-defined actions:
- Customer message routing with 93% accuracy
- Iterative prompt improvement (Prompt 1 → Prompt 2)
- LLM observability with Arize Phoenix
- Evaluation-driven development

**What you'll build:**
- RouterAgent that classifies customer messages into departments
- Evaluation pipeline with 30 test cases
- Phoenix tracing integration for debugging

### V2: Planning Autonomy - Multi-Step Planner
**File:** `v2_planning_autonomy_UPDATED.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agentic_ai_course_lil/v2_planning_autonomy_UPDATED.ipynb)

Build on V1 by adding retrieval and multi-step planning:
- RAG system with BM25 retrieval
- Multi-step action plan generation
- Custom metrics design from observed failures
- LLM-as-Judge evaluation (3-class: good/partial/bad)

**What you'll build:**
- PlanningAgent that generates detailed action plans
- BM25 SOP retrieval system
- Custom evaluation metrics (SOP Recall + Plan Alignment)
- Prompt improvement workflow (K=2→4, gpt-4o→gpt-5)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Jupyter Notebook or Google Colab

### Installation

#### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge above for V1 or V2
2. Add your OpenAI API key to Colab Secrets:
   - Click 🔑 icon in left sidebar
   - Name: `OPENAI_API_KEY`
   - Value: Your API key
   - Enable notebook access
3. Run all cells! (Repository will be cloned automatically)

#### Option 2: Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/aishwaryanr/awesome-generative-ai-guide.git
   cd awesome-generative-ai-guide/resources/agentic_ai_course_lil
   ```
2. Install dependencies:
   ```bash
   pip install openai pandas python-dotenv rank-bm25
   pip install 'arize-phoenix[evals]' openinference-instrumentation-openai
   ```
3. Create `.env` file in the `resources/agentic_ai_course_lil/` directory:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```
4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## 📁 Project Structure

These notebooks are located in the `resources/agentic_ai_course_lil/` directory of the main repository:

```
awesome-generative-ai-guide/
└── resources/
    └── agentic_ai_course_lil/
        ├── v1_action_autonomy.ipynb          # V1: Routing agent
        ├── v2_planning_autonomy_UPDATED.ipynb # V2: Planning agent
        ├── data/
        │   ├── v1_test_cases.csv              # 30 routing test cases
        │   ├── v2_test_cases.csv              # 22 planning test cases
        │   └── sops/                          # Standard Operating Procedures
        │       ├── sop_001.txt                # Returns SOP
        │       ├── sop_003.txt                # Billing SOP
        │       └── ...
        └── assets/
            └── diagrams/                      # Architecture diagrams
                ├── autonomy_ladder.png
                ├── v1_architecture.png
                ├── v1_data_flow.png
                ├── v2_architecture.png
                ├── v2_data_flow.png
                └── v2_sop_retrieval.png
```

## 🎯 Learning Path

**Start here:** V1 Action Autonomy
1. Understand single-action agents
2. Learn evaluation-driven development
3. Use Phoenix for LLM observability
4. Iterate to improve accuracy

**Then:** V2 Planning Autonomy
1. Build on V1's routing (don't start from scratch!)
2. Add retrieval with BM25
3. Generate multi-step plans
4. Design custom metrics from failures

## 🔑 Key Concepts

### V1: Action Autonomy
- **Routing**: Single classification decision
- **Evaluation**: Routing accuracy metric
- **Observability**: Phoenix traces for debugging
- **Iteration**: Prompt 1 (73%) → Prompt 2 (93%)

### V2: Planning Autonomy
- **RAG**: BM25 retrieval of relevant SOPs
- **Planning**: Multi-step action generation
- **Custom Metrics**: SOP Recall + Plan Alignment
- **LLM-as-Judge**: GPT-4o evaluates GPT-5 outputs
- **Trace-First**: Observe → Discover → Measure → Improve

## 📊 Expected Results

### V1 Routing Accuracy
- Prompt 1 (baseline): 73%
- Prompt 2 (improved): 93%
- **Improvement:** +20 percentage points

### V2 Planning Metrics
- **SOP Recall:**
  - Prompt 1 (K=2): 53.79%
  - Prompt 2 (K=4): 75.76% (+40.8%)
- **Plan Alignment:**
  - Prompt 1 (gpt-4o): 72% good plans
  - Prompt 2 (gpt-5): 100% good plans (+28%)

## 🛠️ Technologies Used

- **OpenAI API**: gpt-4o, gpt-4o-mini, gpt-5
- **Arize Phoenix**: LLM observability and tracing
- **BM25 (rank-bm25)**: Keyword-based retrieval
- **Pandas**: Data manipulation
- **Jupyter**: Interactive notebooks

## 📖 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Arize Phoenix Documentation](https://docs.arize.com/phoenix)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

## 🎓 Course Philosophy

1. **Incremental Building**: Each version builds on the previous
2. **Evaluation-First**: Measure before and after improvements
3. **Observe Failures**: Design metrics from real issues, not assumptions
4. **Targeted Improvements**: One change per metric

## 📝 License

MIT License - feel free to use for learning and teaching!

## 🤝 Contributing

This is a teaching resource. If you find issues or have improvements:
1. Open an issue describing the problem
2. Suggest improvements via pull request

## ⚠️ Cost Warning

These notebooks make OpenAI API calls. Costs are typically:
- V1: ~$0.50-$1.00 for full evaluation
- V2: ~$1.00-$2.00 for full evaluation (depends on K and model)

Set up billing limits in your OpenAI account!
