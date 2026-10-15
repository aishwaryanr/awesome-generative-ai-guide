# Agentic AI: Build Your First Agentic AI System

This is the repository for the LinkedIn Learning course `Agentic AI: Build Your First Agentic AI System`. The full course is available from [LinkedIn Learning][lil-course-url].

![Agentic AI: Build Your First Agentic AI System][lil-thumbnail-url]

## Course Description

Learn to build production-ready agentic AI systems using the Autonomy Ladder framework. This hands-on course guides you through implementing two complete systems - from simple action classification to multi-step planning with retrieval - while teaching systematic evaluation, error analysis, and continuous improvement patterns that work in real production environments.

## Instructions

This repository has branches for each of the videos in the course. You can use the branch pop up menu in GitHub to switch to a specific branch and take a look at the course at that stage, or you can add `/tree/BRANCH_NAME` to the URL to go to the branch you want to access.

## Course Notebooks

This course uses two main Jupyter notebooks that you'll run on Google Colab:

### Chapter 3: V1 Action Autonomy (Router Agent)
**[Open action_autonomy.ipynb in Colab](https://colab.research.google.com/github/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agentic_ai_course_lil/action_autonomy.ipynb)** - Use this notebook throughout Chapter 3. The notebook includes clear chapter break markers (🎬 End of Chapter) that show you where to stop for each video.

### Chapter 4: V2 Planning Autonomy (Planning Agent)
**[Open planning_autonomy.ipynb in Colab](https://colab.research.google.com/github/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agentic_ai_course_lil/planning_autonomy.ipynb)** - Use this notebook throughout Chapter 4. The notebook includes clear chapter break markers (🎬 End of Chapter) that show you where to stop for each video.

## Running the Notebooks

All notebooks in this course are designed to run on **Google Colab** with no local setup required.

### Quick Start with Google Colab

1. **Open the notebook in Colab**: Click on the notebook link above (e.g., "Open action_autonomy.ipynb in Colab") to launch it directly in Google Colab

2. **Set up your OpenAI API key**:
   - In Colab, click the key icon (🔑) in the left sidebar
   - Add a new secret named `OPENAI_API_KEY`
   - Paste your OpenAI API key as the value
   - Toggle the "Notebook access" switch to enable access

3. **Run the notebook**: Execute cells in order, stopping at chapter break markers as indicated in the videos

## Prerequisites

- **OpenAI API Key**: You'll need an OpenAI API key to run the notebooks. [Get one here](https://platform.openai.com/api-keys)
- **Basic Python Knowledge**: Familiarity with Python and Jupyter notebooks is helpful
- **Google Account**: Required for using Google Colab

## What You'll Build

### V1: Action Autonomy (Router Agent)
A customer support routing agent that:
- Classifies customer messages into appropriate departments
- Uses systematic evaluation to measure baseline performance
- Applies error analysis to discover improvement opportunities
- Implements targeted improvements and validates results

**Key Concepts:**
- Simple prompt engineering
- Systematic evaluation
- Continuous Calibration (CC): Analyzing failures to discover improvements
- Continuous Deployment (CD): Implementing and validating improvements

### V2: Planning Autonomy (Planning Agent)
A multi-step planning agent that:
- Routes messages using V1's proven routing (builds on V1!)
- Retrieves relevant procedures from a knowledge base using BM25
- Generates detailed, multi-step action plans
- Uses custom metrics to measure and improve retrieval and plan quality

**Key Concepts:**
- RAG (Retrieval Augmented Generation) with BM25
- Custom metrics design from observations
- LLM-as-Judge for plan evaluation
- Incremental building (V2 = V1 + new capabilities)

## Course Structure

The course follows a systematic pattern for both autonomy levels:

1. **Build**: Implement the baseline system
2. **Test**: Run evaluation to establish baseline metrics
3. **Calibrate (CC)**: Observe failures, analyze patterns, design metrics
4. **Deploy (CD)**: Make targeted improvements, re-evaluate, validate gains

This CC/CD pattern works for production systems and teaches you how to systematically improve any agentic AI system.

## Repository Structure

```
├── action_autonomy.ipynb          # Chapter 3: V1 Action Autonomy
├── planning_autonomy.ipynb        # Chapter 4: V2 Planning Autonomy
├── data/
│   ├── v1_test_cases.csv          # Test cases for V1 evaluation
│   ├── v2_test_cases.csv          # Test cases for V2 evaluation
│   └── sops/                      # Standard Operating Procedures (V2)
│       ├── sop_001.txt
│       ├── sop_002.txt
│       └── ...
└── assets/
    └── diagrams/                  # Architecture diagrams
        ├── autonomy_ladder.png
        ├── v1_architecture.png
        ├── v2_architecture.png
        └── ...
```

## Troubleshooting

### "ModuleNotFoundError" in Colab
- Run the first cell that installs packages: `!pip install -q openai pandas ...`
- Restart the runtime if needed: Runtime > Restart runtime

### "Invalid API Key"
- Verify your OpenAI API key is set correctly in Colab Secrets
- Check that "Notebook access" is enabled for the secret

### Phoenix UI not loading
- Use the Colab-compatible Phoenix setup (already configured in notebooks)
- The Phoenix UI may take 30-60 seconds to start

### Images not displaying
- Images are loaded from the `assets/diagrams` folder in the repository
- Ensure you're viewing the notebook from the correct branch

## Additional Resources

- [Arize Phoenix Documentation](https://docs.arize.com/phoenix)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [BM25 Algorithm Explanation](https://en.wikipedia.org/wiki/Okapi_BM25)

## Instructor

**Aishwarya Naresh Reganti**


[lil-course-url]: https://www.linkedin.com/learning/agentic-ai-build-your-first-agentic-ai-system
[lil-thumbnail-url]: https://media.licdn.com/dms/image/v2/D4E0DAQG0eDHsyOSqTA/learning-public-crop_675_1200/B4EZVdqqdwHUAY-/0/1741033220778?e=2147483647&v=beta&t=FxUDo6FA8W8CiFROwqfZKL_mzQhYx9loYLfjN-LNjgA
