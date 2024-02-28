# :star: :bookmark: awesome-generative-ai-guide
Generative AI is experiencing rapid growth, and this repository serves as a comprehensive hub for updates on generative AI research, interview materials, notebooks, and more! 

Explore the following resources:

1. [Monthly Best GenAI Papers List](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#star-best-genai-papers-list-january-2024)
2. [GenAI Interview Resources](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#computer-interview-prep)
3. [Applied LLMs Mastery 2024 (created by Aishwarya Naresh Reganti) course material](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#ongoing-applied-llms-mastery-2024)
4. [List of all GenAI-related free courses (over 30 already listed)](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#book-list-of-free-genai-courses)
5. [List of code repositories/notebooks for developing generative AI applications](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#notebook-code-notebooks)

We'll be updating this repository regularly, so keep an eye out for the latest additions!






Happy Learning!

---
## :speaker: Announcements
- Applied LLMs Mastery full course content has been released!!! [Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024)
- 5-day roadmap to learn LLM foundations out now! ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/genai_roadmap.md))
- 60 Common GenAI Interview Questions out now! ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/interview_prep/60_gen_ai_questions.md))
- ICLR 2024 paper summaries [click here](https://areganti.notion.site/06f0d4fe46a94d62bff2ae001cfec22c?v=d501ca62e4b745768385d698f173ae14)
- List of free GenAI courses [click here](https://github.com/aishwaryanr/awesome-generative-ai-guide#book-list-of-free-genai-courses) 




---

## :star: Best GenAI Papers List (January 2024)
*Updated at the end of every month

| Date        | Name                                                                                                                  | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Topics                  |
| ----------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| 31 Jan 2024 | [Large Language Models for Mathematical Reasoning: Progresses and Challenges](https://arxiv.org/html/2402.00157v1)                                           | This survey delves into the landscape of Large Language Models in mathematical problem-solving, addressing various problem types, datasets, techniques, factors, and challenges. It comprehensively explores the advancements and obstacles in this burgeoning field, providing insights into the current state and future directions. This survey offers a holistic perspective on LLMs' role in mathematical reasoning, aiming to guide future research in this rapidly evolving domain.                                                                                             | Task Specific LLMs      |
| 29 Jan 2024 | [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)                                                                             | To address concerns about retrieval-augmented generation models' robustness, Corrective Retrieval Augmented Generation (CRAG) is proposed. CRAG incorporates a lightweight retrieval evaluator to assess document quality and triggers different retrieval actions based on confidence levels. It extends retrieval results through large-scale web searches and employs a decompose-then-recompose algorithm to focus on key information. Experiments demonstrate CRAG's effectiveness in enhancing RAG-based approaches across various generation tasks.                             | RAG                     |
| 29 Jan 2024 | [MoE-LLaVA: Mixture of Experts for Large Vision-Language Models](https://arxiv.org/abs/2401.15947)                                                         | This work introduces MoE-Tuning, a training strategy for Large Vision-Language Models that addresses the computational costs of existing scaling methods by constructing sparse models with constant computational overhead. It also presents MoE-LLaVA, a MoE-based sparse LVLM architecture that activates only the top-k experts during deployment. Experimental results demonstrate MoE-LLaVA's significant performance across various visual understanding and object hallucination benchmarks, providing insights for more efficient multi-modal learning systems.               | MoE Models              |
| 29 Jan 2024 | [The Power of Noise: Redefining Retrieval for RAG Systems](https://arxiv.org/abs/2401.14887)                                                               | This study examines the impact of Information Retrieval components on Retrieval-Augmented Generation systems, complementing previous research focused on LLMs' generative aspect within RAG systems. By analyzing characteristics such as document relevance, position, and context size, the study reveals unexpected insights, like the surprising performance boost from including irrelevant documents. These findings emphasize the importance of developing specialized strategies to integrate retrieval with language generation models, guiding future research in this area. | RAG                     |
| 24 Jan 2024 | [MM-LLMs: Recent Advances in MultiModal Large Language Models](https://arxiv.org/abs/2401.13601)                                                           | This paper presents a comprehensive survey of MultiModal Large Language Models (MM-LLMs), which augment off-the-shelf LLMs to support multimodal inputs or outputs. It outlines design formulations, introduces 26 existing MM-LLMs, reviews their performance on mainstream benchmarks, and summarizes key training recipes. Promising directions for MM-LLMs are explored, alongside a real-time tracking website for the latest developments, aiming to contribute to the ongoing advancement of the MM-LLMs domain.                                                                | Multimodal LLMs         |
| 23 Jan 2024 | [Red Teaming Visual Language Models](https://arxiv.org/abs/2401.12915)                                                                                     | A novel red teaming dataset, RTVLM, is introduced to assess Vision-Language Models' (VLMs) performance in generating harmful or inaccurate content. It encompasses 10 subtasks across faithfulness, privacy, safety, and fairness aspects. Analysis reveals significant performance gaps among prominent open-source VLMs, prompting exploration of red teaming alignment techniques. Application of red teaming alignment to LLaVA-v1.5 bolsters model performance, indicating the need for further development in this area.                                                         | Red-Teaming             |
| 23 Jan 2024 | [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945)                                                            | Lumiere is introduced as a text-to-video diffusion model aimed at synthesizing realistic and coherent motion in videos. It employs a Space-Time U-Net architecture to generate entire video durations in a single pass, enabling global temporal consistency. Through spatial and temporal down- and up-sampling, and leveraging a pre-trained text-to-image diffusion model, Lumiere achieves state-of-the-art text-to-video generation results, facilitating various content creation and video editing tasks with ease.                                                             | Diffusion Models        |
| 22 Jan 2024 | [WARM: On the Benefits of Weight Averaged Reward Models](https://arxiv.org/abs/2401.12187)                                                                | Reinforcement Learning with Human Feedback for large language models can lead to reward hacking. To address this, Weight Averaged Reward Models (WARM) are proposed, where multiple fine-tuned reward models are averaged in weight space. WARM improves efficiency and reliability under distribution shifts and preference inconsistencies, enhancing the quality and alignment of LLM predictions. Experiments on summarization tasks demonstrate WARM's effectiveness, with RL fine-tuned models using WARM outperforming single RM counterparts.                                 | Instruction Tuning      |
| 18 Jan 2024 | [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)                                                                                        | This paper introduces Self-Rewarding Language Models, where the language model itself provides rewards during training via LLM-as-a-Judge prompting. Through iterative training, the model not only improves its instruction-following ability but also enhances its capacity to generate high-quality rewards. Fine-tuning Llama 2 70B using this approach yields a model that surpasses existing systems on the AlpacaEval 2.0 leaderboard, showcasing potential for continual improvement in both performance axes.                                                                 | Prompt Engineering      |
| 16 Jan 2024 | [Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering](https://arxiv.org/abs/2401.08500)                                        | AlphaCodium is proposed as a new approach to code generation by Large Language Models, emphasizing a test-based, multi-stage, code-oriented iterative flow tailored for code tasks. Tested on the CodeContests dataset, AlphaCodium consistently improves LLM performance, significantly boosting accuracy compared to direct prompts. The principles and best practices derived from this approach are deemed broadly applicable to general code generation tasks.                                                                                                                    | Code Generation         |
| 13 Jan 2024 | [Leveraging Large Language Models for NLG Evaluation: A Survey](https://arxiv.org/html/2401.07103v1)                                                        | This survey delves into leveraging Large Language Models for evaluating Natural Language Generation, providing a comprehensive taxonomy for organizing existing evaluation metrics. It critically assesses LLM-based methodologies, highlighting their strengths, limitations, and unresolved challenges such as bias and domain-specificity. The survey aims to offer insights to researchers and advocate for fairer and more advanced NLG evaluation techniques.                                                                                                                    | Evaluation              |
| 12 Jan 2024 | [How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs](https://arxiv.org/html/2401.06373v1)      | This paper explores a new perspective on AI safety by considering large language models as human-like communicators and studying how to jailbreak them through persuasion. It introduces a persuasion taxonomy and applies it to generate interpretable persuasive adversarial prompts (PAP), achieving high attack success rates on LLMs like GPT-3.5 and GPT-4. The study also highlights gaps in existing defenses against such attacks and advocates for more fundamental mitigation strategies for interactive LLMs.                                                              | Red-Teaming             |
| 11 Jan 2024 | [Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856)                                      | The paper explores the integration of semantic search capabilities into applications through Retrieval Augmented Generation (RAG) systems. It identifies seven failure points in RAG system design based on case studies across various domains. Key takeaways include the feasibility of validating RAG systems during operation and the evolving nature of system robustness. The paper concludes with suggestions for potential research directions to enhance RAG system effectiveness. | RAG                     |
| 10 Jan 2024 | [TrustLLM: Trustworthiness in Large Language Models](https://arxiv.org/abs/2401.05561)                                                                     | The paper examines trustworthiness in large language models like ChatGPT, proposing principles and benchmarks. It evaluates 16 LLMs, finding a correlation between trustworthiness and effectiveness, but noting concerns about proprietary models outperforming open-source ones. It emphasizes the need for transparency in both models and underlying technologies for trustworthiness analysis.                                                                                                                                                                                    | Alignment               |
| 9 Jan 2024  | [Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding](https://arxiv.org/abs/2401.04398)                                          | The Chain-of-Table framework proposes leveraging tabular data explicitly in the reasoning chain to enhance table-based reasoning tasks. It guides large language models using in-context learning to iteratively generate operations and update the table, allowing for dynamic planning based on previous results. This approach achieves state-of-the-art performance on various table understanding benchmarks, showcasing its effectiveness in enhancing LLM-based reasoning.                                                                                                      | Prompt Engineering, RAG |
| 8 Jan 2024  | [MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts](https://arxiv.org/abs/2401.04081)                                              | The paper introduces MoE-Mamba, a model combining Mixture of Experts (MoE) with Sequential State Space Models (SSMs) to enhance scaling and performance. MoE-Mamba surpasses both Mamba and Transformer-MoE, achieving Transformer-like performance with fewer training steps while maintaining the inference gains of Mamba over Transformers.                                                                                                                                                                                                                                        | MoE Models              |
| 4 Jan 2024  | [Blending Is All You Need: Cheaper, Better Alternative to Trillion-Parameters LLM](https://arxiv.org/abs/2401.02994)                                      | The paper investigates whether combining smaller chat AI models can match or exceed the performance of a single large model like ChatGPT, without requiring extensive computational resources. Through empirical evidence and A/B testing on the Chai research platform, the "blending" approach demonstrates potential to rival or surpass the capabilities of larger models.                                                                                                                                                                                                          | Smaller Models          |

---




## :mortar_board: Courses
#### [Ongoing] Applied LLMs Mastery 2024
Join 1000+ students on this 10-week adventure as we delve into the application of LLMs across a variety of use cases
#### [Link](https://areganti.notion.site/Applied-LLMs-Mastery-2024-562ddaa27791463e9a1286199325045c) to the course website
##### [Feb 2024] Registrations are still open [click here](https://forms.gle/353sQMRvS951jDYu7) to register

🗓️*Week 1 [Jan 15 2024]***: [Practical Introduction to LLMs](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week1_part1_foundations.md)**
- Applied LLM Foundations
- Real World LLM Use Cases
- Domain and Task Adaptation Methods

🗓️*Week 2 [Jan 22 2024]***: [Prompting and Prompt 
Engineering](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week2_prompting.md)**
- Basic Prompting Principles
- Types of Prompting
- Applications, Risks and Advanced Prompting

🗓️*Week 3 [Jan 29 2024]***: [LLM Fine-tuning](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week3_finetuning_llms.md)** 
- Basics of Fine-Tuning
- Types of Fine-Tuning
- Fine-Tuning Challenges

🗓️*Week 4 [Feb 5 2024]***: [RAG (Retrieval-Augmented Generation)](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week4_RAG.md)**
- Understanding the concept of RAG in LLMs
- Key components of RAG
- Advanced RAG Methods

🗓️*Week 5 [ Feb 12 2024]***: [Tools for building LLM Apps](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week5_tools_for_LLM_apps.md)**
- Fine-tuning Tools
- RAG Tools
- Tools for observability, prompting, serving, vector search etc.

🗓️*Week 6 [Feb 19 2024]***: [Evaluation Techniques](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week6_llm_evaluation.md)**
- Types of Evaluation
- Common Evaluation Benchmarks
- Common Metrics

🗓️*Week 7 [Feb 26 2024]***: [Building Your Own LLM Application](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week7_build_llm_app.md)**
- Components of LLM application
- Build your own LLM App end to end

🗓️*Week 8 [March 4 2024]***: [Advanced Features and Deployment](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week8_advanced_features.md)**
- LLM lifecycle and LLMOps
- LLM Monitoring and Observability
- Deployment strategies

🗓️*Week 9 [March 11 2024]***: [Challenges with LLMs](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week9_challenges_with_llms.md)**
- Scaling Challenges
- Behavioral Challenges
- Future directions

🗓️*Week 10 [March 18 2024]***: [Emerging Research Trends](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week10_research_trends.md)**
- Smaller and more performant models
- Multimodal models
- LLM Alignment

🗓️*Week 11 *Bonus* [March 25 2024]***: [Foundations](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week11_foundations.md)**
- Generative Models Foundations
- Self-Attention and Transformers
- Neural Networks for Language

---

#### :book: List of Free GenAI Courses
##### LLM Basics and Foundations

1. [Large Language Models](https://rycolab.io/classes/llm-s23/) by ETH Zurich

2. [Understanding Large Language Models](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/) by Princeton

3. [Transformers course](https://huggingface.co/learn/nlp-course/chapter1/1) by Huggingface

4. [NLP course](https://huggingface.co/learn/nlp-course/chapter1/1) by Huggingface

5. [CS324 - Large Language Models](https://stanford-cs324.github.io/winter2022/) by Stanford

6. [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) by Coursera

7. [Introduction to Generative AI](https://www.coursera.org/learn/introduction-to-generative-ai) by Coursera

8. [Generative AI Fundamentals](https://www.cloudskillsboost.google/paths/118/course_templates/556) by Google Cloud

9. [Introduction to Large Language Models](https://www.cloudskillsboost.google/paths/118/course_templates/539) by Google Cloud
11. [Introduction to Generative AI](https://www.cloudskillsboost.google/paths/118/course_templates/536) by Google Cloud
12. [Generative AI Concepts](https://www.datacamp.com/courses/generative-ai-concepts) by DataCamp (Daniel Tedesco Data Lead @ Google)
13. [1 Hour Introduction to LLM (Large Language Models)](https://www.youtube.com/watch?v=xu5_kka-suc) by WeCloudData



##### Building LLM Applications

1. [LLMOps: Building Real-World Applications With Large Language Models](https://www.udacity.com/course/building-real-world-applications-with-large-language-models--cd13455) by Udacity

2. [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) by FSDL

3. [Generative AI for beginners](https://github.com/microsoft/generative-ai-for-beginners/tree/main) by Microsoft

4. [Large Language Models: Application through Production](https://www.edx.org/learn/computer-science/databricks-large-language-models-application-through-production) by Databricks

5. [Generative AI Foundations](https://www.youtube.com/watch?v=oYm66fHqHUM&list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF) by AWS

6. [Introduction to Generative AI Community Course](https://www.youtube.com/watch?v=ajWheP8ZD70&list=PLmQAMKHKeLZ-iTT-E2kK9uePrJ1Xua9VL) by ineuron

7. [LLM University](https://docs.cohere.com/docs/llmu) by Cohere
8. [LLM Learning Lab](https://lightning.ai/pages/llm-learning-lab/) by Lightning AI

9. [Functions, Tools and Agents with LangChain](https://learn.deeplearning.ai/functions-tools-agents-langchain) by Deeplearning.AI

10. [LangChain for LLM Application Development](https://learn.deeplearning.ai/login?redirect_course=langchain&callbackUrl=https%3A%2F%2Flearn.deeplearning.ai%2Fcourses%2Flangchain) by Deeplearning.AI

11. [LLMOps](https://learn.deeplearning.ai/llmops) by DeepLearning.AI

12. [Automated Testing for LLMOps](https://learn.deeplearning.ai/automated-testing-llmops) by DeepLearning.AI

##### 3. Prompt Engineering, RAG and Fine-Tuning
1. [LangChain & Vector Databases in Production](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVhnQW8xNDdhSU9IUDVLXzFhV2N0UkNRMkZrQXxBQ3Jtc0traUxHMzZJcGJQYjlyckYxaGxYVWlsOFNGUFlFVEdhNzdjTWpPUlQ2TF9XczRqNkxMVGpJTnd5YmYzV0prQ0IwZURNcHhIZ3h1Z051VTl5MXBBLUN0dkM0NHRkQTFua1Jpc0VCRFJUb0ZQZG95b0JqMA&q=https%3A%2F%2Flearn.activeloop.ai%2Fcourses%2Flangchain&v=gKUTDC13jys) by Activeloop

2. [Reinforcement Learning from Human Feedback](https://learn.deeplearning.ai/reinforcement-learning-from-human-feedback) by DeepLearning.AI

3. [Building Applications with Vector Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by DeepLearning.AI

4. [How Diffusion Models Work](https://learn.deeplearning.ai/diffusion-models) by DeepLearning.AI
5. [Finetuning Large Language Models](https://learn.deeplearning.ai/finetuning-large-language-models) by Deeplearning.AI
6. [LangChain: Chat with Your Data](http://learn.deeplearning.ai/langchain-chat-with-your-data/) by Deeplearning.AI

7. [Building Systems with the ChatGPT API](https://learn.deeplearning.ai/chatgpt-building-system) by Deeplearning.AI
8. [Building Applications with Vector Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by Deeplearning.AI
9. [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction) by Deeplearning.AI
10. [Advanced RAG Orchestration series](https://www.youtube.com/watch?v=CeDS1yvw9E4) by LlamaIndex

##### Evaluation
1. [Building and Evaluating Advanced RAG Applications](https://learn.deeplearning.ai/building-evaluating-advanced-rag) by DeepLearning.AI

2. [Evaluating and Debugging Generative AI Models Using Weights and Biases](https://learn.deeplearning.ai/evaluating-debugging-generative-ai) by Deeplearning.AI



---

## :paperclip: Resources
- [ICLR 2024 Paper Summaries](https://areganti.notion.site/06f0d4fe46a94d62bff2ae001cfec22c?v=d501ca62e4b745768385d698f173ae14)


---


## :computer: Interview Prep 
#### Topic wise Questions:
1. [Common GenAI Interview Questions](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/interview_prep/60_gen_ai_questions.md)
2. Prompting and Prompt Engineering 
3. Model Fine-Tuning 
4. Model Evaluation 
5. MLOps for GenAI 
6. Generative Models Foundations 
7. Latest Research Trends 

#### GenAI System Design (Coming Soon):

1. Designing an LLM-Powered Search Engine
2. Building a Customer Support Chatbot
3. Building a system for natural language interaction with your data.
4. Building an AI Co-pilot
5. Designing a Custom Chatbot for Q/A on Multimodal Data (Text, Images, Tables, CSV Files)
6. Building an Automated Product Description and Image Generation System for E-commerce



---
## :notebook: Code Notebooks
#### RAG Tutorials
- [AWS Bedrock Workshop Tutorials](https://github.com/aws-samples/amazon-bedrock-workshop) by Amazon Web Services
- [Langchain Tutorials](https://github.com/gkamradt/langchain-tutorials) by gkamradt
- [LLM Applications for production](https://github.com/ray-project/llm-applications/tree/main) by ray-project
- [LLM tutorials](https://github.com/ollama/ollama/tree/main/examples) by Ollama
- [LLM Hub](https://github.com/mallahyari/llm-hub) by mallahyari

#### Fine-Tuning Tutorials
- [LLM Fine-tuning tutorials](https://github.com/ashishpatel26/LLM-Finetuning) by ashishpatel26
- [PEFT](https://github.com/huggingface/peft/tree/main/examples) example notebooks by Huggingface
- [Free LLM Fine-Tuning Notebooks](https://levelup.gitconnected.com/14-free-large-language-models-fine-tuning-notebooks-532055717cb7) by Youssef Hosni



---

## :black_nib: Contributing
If you want to add to the repository or find any issues, please feel free to raise a PR and ensure correct placement within the relevant section or category.


---

## :pushpin: Cite Us

To cite this guide, use the below format:

```
@article{areganti_generative_ai_guide,
author = {Reganti, Aishwarya Naresh},
journal = {https://github.com/aishwaryanr/awesome-generative-ai-resources},
month = {01},
title = {{Generative AI Guide}},
year = {2024}
}
```

## License

[MIT License]


