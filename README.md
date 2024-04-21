# :star: :bookmark: awesome-generative-ai-guide
Generative AI is experiencing rapid growth, and this repository serves as a comprehensive hub for updates on generative AI research, interview materials, notebooks, and more! 

Explore the following resources:

1. [Monthly Best GenAI Papers List](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#star-best-genai-papers-list-january-2024)
2. [GenAI Interview Resources](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#computer-interview-prep)
3. [Applied LLMs Mastery 2024 (created by Aishwarya Naresh Reganti) course material](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#ongoing-applied-llms-mastery-2024)
4. [List of all GenAI-related free courses (over 70 listed)](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#book-list-of-free-genai-courses)
5. [List of code repositories/notebooks for developing generative AI applications](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#notebook-code-notebooks)

We'll be updating this repository regularly, so keep an eye out for the latest additions!






Happy Learning!

---
## :speaker: Announcements
- Applied LLMs Mastery full course content has been released!!! ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024))
- 5-day roadmap to learn LLM foundations out now! ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/genai_roadmap.md))
- 60 Common GenAI Interview Questions out now! ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/interview_prep/60_gen_ai_questions.md))
- ICLR 2024 paper summaries ([Click Here](https://areganti.notion.site/06f0d4fe46a94d62bff2ae001cfec22c?v=d501ca62e4b745768385d698f173ae14))
- List of free GenAI courses ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide#book-list-of-free-genai-courses))
- Generative AI resources and roadmaps
  - [3-day RAG roadmap](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/RAG_roadmap.md)
  - [5-day LLM foundations roadmap](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/genai_roadmap.md)
  - [5-day LLM agents roadmap](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agents_roadmap.md)
  - [Agents 101 guide](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agents_101_guide.md)




---

## :star: Best GenAI Papers List (March 2024)
*Updated at the end of every month

| Date | Name | Summary | Topics |
|-------|-------------|------|------|
| 29 March 2024 |[Gecko: Versatile Text Embeddings Distilled from Large Language Models](https://arxiv.org/abs/2403.20327) | Gecko introduces a novel approach for creating compact and efficient text embeddings by distilling knowledge from large language models into a retriever. Utilizing a two-step distillation process that generates diverse, synthetic paired data, Gecko achieves superior retrieval performance. With a focus on compactness, it outperforms larger models and higher-dimensional embeddings on the Massive Text Embedding Benchmark (MTEB), demonstrating its efficacy and potential in improving information retrieval tasks. | LLM Embeddings|
| 28 March 2024 |[Grok-1.5](https://x.ai/blog/grok-1.5) | Grok 1.5 offers enhanced reasoning capabilities and a context length of 128,000 tokens. It showcases significant advancements in coding, math-related tasks, and long context understanding. With improvements in MATH, GSM8K, and HumanEval benchmarks, Grok-1.5 offers expanded memory capacity and exceptional retrieval capabilities. Built on a custom distributed training framework, it promises efficiency and reliability for large-scale language model research | Foundational LLM|
| 28 March 2024 |[Don't Use Your Data All at Once: sDPO](https://arxiv.org/abs/2403.19270) | sDPO introduces a novel method in the realm of language model training, focusing on the strategic use of preference datasets in a stepwise manner. This technique enhances model alignment with human preferences by employing parts of the dataset progressively, leading to more precise reference models and outperforming other popular LLMs in terms of performance, even those with more parameters. | Instruction Tuning|
| 28 March 2024 |[Jamba: AI21's SSM-Transformer Model](https://www.ai21.com/blog/announcing-jamba) | AI21 labs announced Jamba novel SSM-Transformer model offering a 256K context window, aiming to balance the SSM model's efficiency with the Transformer's capability. It shows significant performance improvements across various benchmarks. Jamba is open-source under Apache 2.0, available on Hugging Face, and soon on NVIDIA's API catalog, marking a significant advancement in hybrid model architecture​ | Foundational LLM|
| 28 March 2024 |[STaR-GATE: Teaching Language Models to Ask Clarifying Questions](https://arxiv.org/abs/2403.19154) | This paper presents STaR-GATE, a novel approach for enhancing language models' interaction skills by training them to ask clarifying questions. By employing a strategic teacher-student learning framework, STaR-GATE aims to improve the models' ability to clarify ambiguities in user queries, thereby enhancing communication effectiveness and accuracy in understanding and responding to complex requests | Prompt Engineering|
| 27 March 2024 |[Long-form factuality in large language models ](https://arxiv.org/abs/2403.18802v1) | This paper tackles the challenge of factuality in LLM-generated content on open-ended topics. It introduces LongFact, a set of prompts for evaluating long-form factuality, and proposes the Search-Augmented Factuality Evaluator (SAFE) method. SAFE assesses the accuracy of facts in LLM responses through a multi-step reasoning process, comparing supported facts against Google Search results. The findings indicate LLMs' potential for superhuman factuality assessment, offering a cost-effective alternative to human annotation. | LLM Factuality|
| 27 March 2024 |[Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models](https://arxiv.org/abs/2403.18814v1) | Mini-Gemini presents a framework to enhance multi-modal Vision Language Models (VLMs) by improving visual tokens, constructing high-quality datasets, and guiding VLM-based generation for better performance. It uses an additional visual encoder for high-resolution refinement without increasing visual token count, aiming to enhance image understanding, reasoning, and simultaneous generation capabilities of VLMs. Mini-Gemini has shown leading performance in zero-shot benchmarks, surpassing developed private models. | Multimodal LLM|
| 27 March 2024 |[DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) | A state-of-the-art open large language model surpassing established models like GPT-3.5 and competing with Gemini 1.0 Pro. DBRX excels in programming and general LLM capabilities, featuring a fine-grained mixture-of-experts architecture for enhanced training and inference efficiency. It's 40% the size of Grok-1, offering faster inference and reduced compute requirements. The model is available on Hugging Face, emphasizing Databricks' commitment to open models and enabling customers to pretrain DBRX-class models with their infrastructure | Foundational LLM|
| 25 March 2024 |[AIOS: LLM Agent Operating System ](https://arxiv.org/abs/2403.16971) | AIOS is designed as an LLM agent operating system to optimize resource allocation, enable concurrent execution, and provide access control. It embeds LLMs into operating systems, presenting an "OS with soul" toward AGI. The system improves the performance and efficiency of LLM agents, offering a pioneering platform for the AIOS ecosystem development. | Agents|
| 22 March 2024 |[RankPrompt: Step-by-Step Comparisons Make Language Models Better Reasoners](https://arxiv.org/abs/2403.12373) | The paper introduces RankPrompt, a novel prompting method aimed at improving the reasoning capabilities of Large Language Models like ChatGPT and GPT-4. Unlike existing solutions requiring human annotations or failing in inconsistent scenarios, RankPrompt enables LLMs to self-rank their responses by comparing diverse outputs. Experiments across 11 reasoning tasks demonstrate significant performance enhancements, with up to a 13% improvement. Moreover, RankPrompt aligns with human judgments 74% of the time in open-ended evaluations and exhibits robustness to response variations. This method proves effective in eliciting high-quality feedback from LLMs, offering promising avenues for advancing reasoning abilities. | Prompt Engineering|
| 22 March 2024 |[Mora: Enabling Generalist Video Generation via A Multi-Agent Framework](https://arxiv.org/abs/2403.13248) | Mora proposes a new multi-agent framework to address the gap in generalist video generation capabilities, aiming to match the performance of the pioneering model Sora. It leverages multiple visual AI agents to achieve text-to-video generation, image-to-video conversion, video extension, editing, connection, and digital world simulation, demonstrating close performance to Sora across various tasks but with a noticeable gap when assessed holistically. | Multimodal LLM|
| 22 March 2024 |[LLM2LLM: Boosting LLMs with Novel Iterative Data Enhancement](https://arxiv.org/abs/2403.15042) | This study introduces LLM2LLM, a data enhancement strategy utilizing a teacher-student LLM framework for improving performance in tasks with limited data. It involves fine-tuning a student LLM on initial seed data, identifying errors, and generating new data based on these errors using a teacher LLM. This iterative process significantly boosts LLM performance in low-data regimes across various datasets, demonstrating substantial improvements over traditional fine-tuning and other augmentation methods. | Data Augmentation|
| 21 March 2024 |[Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/pdf/2403.14403.pdf) | The paper introduces a novel adaptive QA framework. It dynamically selects the most appropriate strategy for handling queries of varying complexities, from simple to sophisticated, by integrating retrieval-augmented LLMs with a complexity-level classifier. This approach aims to balance efficiency and accuracy in response generation across different query types, showing improvements over existing models and adaptive retrieval methods | RAG|
| 20 March 2024 |[Evaluating Frontier Models for Dangerous Capabilities](https://arxiv.org/abs/2403.13793) | This paper pioneers "dangerous capability" evaluations, focusing on areas like persuasion, cyber-security, self-proliferation, and self-reasoning, using Gemini 1.0 models. While no strong dangerous capabilities were found, early warning signs were identified. The study aims to advance the science of evaluating such capabilities in AI models, preparing for future advancements. | LLM Attacks|
| 19 March 2024 |[Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/abs/2403.13187) | This paper presents an new approach for automating the creation of powerful foundation models by merging diverse open-source models. It optimizes beyond individual model weights, facilitating cross-domain merging and achieving state-of-the-art performance, notably in Japanese language tasks. This approach introduces a new paradigm for automated model composition, offering efficient alternatives for foundation model development. | Model Merging|
| 19 March 2024 |[Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models](https://arxiv.org/abs/2403.12881v1) | This paper addresses the challenge of integrating agent abilities into Large Language Models  for improved performance in NLP tasks. It identifies key observations regarding the entanglement of agent training data, varying learning speeds of LLMs, and side-effects of existing approaches. Introducing Agent-FLAN, a method for Fine-tuning LANguage models for Agents, the paper proposes a novel approach to address these challenges. By carefully redesigning the training corpus and incorporating negative samples, Agent-FLAN enables significant performance improvements, outperforming prior works by 3.5% across multiple evaluation datasets. Moreover, it mitigates hallucination issues and enhances LLMs' agent capabilities, even with scaled model sizes, while slightly improving their general capability. | Agents, Hallucination|
| 18 March 2024 |[What Are Tools Anyway? A Survey from the Language Model Perspective](https://zorazrw.github.io/files/WhatAreToolsAnyway.pdf) | This paper dives into the role of tools in enhancing the performance of language models for text generation tasks. It addresses the ambiguity surrounding the term "tool" and explores how tools aid LMs. Through a systematic review, the paper defines tools as external programs utilized by LMs and examines different tooling scenarios and approaches. Empirical studies assess the efficiency of various tooling methods by measuring compute requirements and performance gains across benchmarks. The survey also identifies challenges and potential avenues for future research in LM tooling. | Agents, Tools, Survey|
| 17 March 2024 |[Grok-1](https://x.ai/blog/grok/model-card) | Grok-1 is an autoregressive Transformer-based model designed for next-token prediction, fine-tuned with feedback from Grok-0 models and humans. Released in November 2023, it boasts a context length of 8,192 tokens and is geared towards various NLP tasks like question answering and coding assistance. However, while Grok-1 excels in information processing, human review is essential to ensure accuracy as it lacks independent web-search capabilities. Despite access to external sources, the model may still hallucinate. Trained on data up to Q3 2023 from the internet and AI Tutors, its performance was evaluated on reasoning tasks and foreign math questions, with ongoing testing involving early adopters for further refinement. | Foundational LLM|
| 15 March 2024 |[RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131) | This paper introduces Retrieval Augmented FineTuning (RAFT), a training approach aimed at enhancing the ability of Large Language Models  to answer questions in domain-specific settings. RAFT leverages retrieval augmented fine-tuning to enable the model to effectively incorporate new knowledge into its reasoning process. By training the model to disregard irrelevant documents (distractor documents) and cite relevant sequences from retrieved documents, RAFT improves the model's ability to provide accurate and coherent responses. Experimental results across various datasets demonstrate the effectiveness of RAFT in domain-specific Retrieval Augmented Generation, offering a valuable post-training recipe for enhancing pre-trained LLMs in domain-specific contexts. | RAG, Fine-Tuning|
| 14 March 2024 |[Logits of API-Protected LLMs Leak Proprietary Information](https://arxiv.org/abs/2403.09539) | This paper reveals that even with restricted API access to proprietary Large Language Models, significant proprietary information can be inferred from a small number of API queries. By exploiting a softmax bottleneck present in most modern LLMs, the research demonstrates the ability to unveil hidden aspects of the model architecture and obtain full-vocabulary outputs. This includes efficiently discovering hidden model sizes, identifying different model updates, and estimating output layer parameters. Empirical investigations on OpenAI's gpt-3.5-turbo reveal its embedding size to be approximately 4,096. The paper concludes by discussing potential measures for LLM providers to mitigate such attacks and suggests viewing these capabilities as opportunities for enhanced transparency and accountability rather than vulnerabilities. | LLM Attacks, Privacy|
| 14 March 2024 |[Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) | This paper introduces Quiet-STaR, a method aimed at enabling language models to learn to generate rationales to explain future text, thereby improving their predictive abilities. Building upon the Self-Taught Reasoner (STaR) framework, Quiet-STaR allows LMs to infer unstated rationales in arbitrary text. Key challenges addressed include computational costs, LM's initial unfamiliarity with generating internal thoughts, and predicting beyond individual tokens. The proposed method involves tokenwise parallel sampling, learnable tokens for indicating thought boundaries, and extended teacher-forcing techniques. Quiet-STaR leads to significant improvements in LM performance on tasks like GSM8K and CommonsenseQA without requiring fine-tuning, marking a step towards more general and scalable reasoning capabilities in LMs. | Prompt Engineering|
| 14 March 2024 |[MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611) | This paper explores the development of high-performing Multimodal Large Language Models (MLLMs) and investigates the significance of various architecture components and data choices. Through meticulous ablations of the image encoder, vision language connector, and pre-training data options, several crucial design insights are uncovered. For instance, the careful integration of image-caption, interleaved image-text, and text-only data is shown to be essential for achieving state-of-the-art few-shot results across multiple benchmarks. Additionally, the impact of image resolution and token count in the image encoder is highlighted, while the vision-language connector design is found to be comparatively less critical. Scaling up the proposed approach results in MM1, a family of multimodal models with up to 30B parameters, including dense models and mixture-of-experts variants. MM1 achieves state-of-the-art pre-training metrics and competitive performance on various multimodal benchmarks, benefiting from enhanced in-context learning and multi-image reasoning capabilities enabled by large-scale pre-training. | Multimodal LLM|
| 13 March 2024 |[Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/abs/2403.08319) | This survey dives into the intricacies of knowledge conflicts encountered by large language models, focusing on the blending of contextual and parametric knowledge. It identifies three main categories of conflicts: context-memory, inter-context, and intra-memory conflicts, which can significantly impact LLM trustworthiness and performance, particularly in real-world scenarios with noise and misinformation. Through categorization, exploration of causes, observation of LLM behaviors, and review of existing solutions, the survey aims to provide insights into strategies for enhancing LLM robustness, serving as a valuable resource for advancing research in this domain. | LLM Robustness|
| 12 March 2024 |[MoAI: Mixture of All Intelligence for Large Language and Vision Models](https://arxiv.org/abs/2403.07508) | MoAI introduces an innovative approach to combine the strengths of large language and vision models with specialized computer vision models for tasks like segmentation and OCR. By leveraging auxiliary visual information and blending it with language features through a unique modular design, MoAI achieves superior performance in various zero-shot visual language tasks, particularly in real-world scene understanding, without increasing model size or requiring additional visual instruction datasets.| Multimodal LLMs|
| 12 March 2024 |[Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM](https://arxiv.org/abs/2403.07816) | This paper explores methods for efficiently training Large Language Models  to excel in multiple specialized domains such as coding, math reasoning, and world knowledge. Introducing Branch-Train-MiX (BTX), the approach starts with a seed model and branches to train experts in parallel, reducing communication costs. After training, BTX combines the experts' feedforward parameters into Mixture-of-Expert (MoE) layers, followed by an MoE-finetuning stage to learn token-level routing. BTX encompasses two special cases: Branch-Train-Merge, which lacks the MoE finetuning stage, and sparse upcycling, which skips asynchronous training. Results demonstrate that BTX offers the best accuracy-efficiency tradeoff compared to alternative methods. | MoEs, Foundational LLM|
| 11 March 2024 |[Stealing Part of a Production Language Model](https://arxiv.org/abs/2403.06634) | This paper presents the first model-stealing attack capable of extracting precise information from black-box production language models like OpenAI's ChatGPT or Google's PaLM-2. By leveraging typical API access, the attack can recover the embedding projection layer of a transformer model, including symmetries. Remarkably, the attack achieves this for under $20 USD, revealing hidden dimensions of 1024 and 2048 for OpenAI's Ada and Babbage models, respectively. Additionally, the exact hidden dimension size of the gpt-3.5-turbo model is recovered, with an estimated cost of under $2,000 in queries to retrieve the entire projection matrix. The paper concludes with discussions on potential defenses and mitigations, as well as implications for future work that could extend the attack. | LLM Attacks|
| 8 March 2024 |[RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation](nan) | This paper introduces Retrieval Augmented Thoughts (RAT), a method aimed at enhancing large language models' reasoning and generation abilities in long-horizon generation tasks while reducing hallucination. RAT iteratively revises a chain of thoughts by incorporating relevant retrieved information at each step. Applied to GPT-3.5, GPT-4, and CodeLLaMA-7b, RAT significantly improves performance across various tasks, with average rating score increases of 13.63% in code generation, 16.96% in mathematical reasoning, 19.2% in creative writing, and 42.78% in embodied task planning. | RAG, Prompt Engineering|
| 7 March 2024 |[Common 7B Language Models Already Possess Strong Math Capabilities](https://arxiv.org/abs/2403.05313) | This research reveals that smaller, 7B-sized language models, specifically LLaMA-2, already exhibit strong mathematical abilities, challenging previous assumptions that such capabilities require very large models or extensive math-focused pre-training. By leveraging synthetic data and scaling strategies, the study significantly improves the model's math-solving accuracy, surpassing previous benchmarks and demonstrating that with appropriate training, even relatively small models can achieve remarkable math performance. | Domain Specific LLMs|
| 7 March 2024 |[ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853) | This paper introduces ShortGPT, which demonstrates a high degree of redundancy across the layers of large language models. By evaluating the necessity of each layer through a metric called Block Influence (BI), the authors propose a straightforward pruning method. Their approach, which simplifies the model by removing redundant layers, shows significant improvements in efficiency without compromising on the model's performance, marking a step forward in optimizing LLM architectures.| Smaller LLMs|
| 7 March 2024 |[Can Large Language Models Reason and Plan?](https://arxiv.org/abs/2403.04121) | This paper questions the ability of large language models to perform self-critique and correct their erroneous guesses, a capability humans occasionally demonstrate. This inquiry underscores the distinct nature of human cognitive processes compared to the computational mechanisms of LLMs, challenging the assumption of equivalent reasoning and self-correction abilities between the two. | Prompt Engineering|
| 6 March 2024 |[GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) | This paper proposes a novel training strategy called GaLore. This approach aims to reduce the memory requirements of training large language models by implementing gradient low-rank projection, significantly cutting down the memory used by optimizer states without sacrificing performance. It allows for the efficient training of large models on consumer-grade GPUs, marking a significant advancement in the accessibility of AI model training. | Memory Optimization|
| 5 March 2024 |[KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents](https://arxiv.org/abs/2403.03101) | The work introduces KnowAgent, a novel approach designed to enhance large language models' planning capabilities by incorporating explicit action knowledge. This integration aims to address the inadequacies in current models that lack built-in action knowledge, leading to planning hallucination. KnowAgent uses an action knowledge base and a self-learning strategy to guide planning trajectories, resulting in more accurate and efficient problem-solving across various domains. | Agents|
| 4 March 2024 |[The Claude 3 Model Family: Opus, Sonnet, Haiku](nan) | This technical report from Claude introduces Claude 3, a new family of large multimodal models designed to address various needs within the AI landscape. Claude 3 comprises three distinct offerings: Opus, Sonnet, and Haiku, each tailored to different requirements in terms of capability, speed, and cost-effectiveness. All models feature vision capabilities for image data processing. Across benchmark evaluations, the Claude 3 family demonstrates robust performance, setting new standards in reasoning, math, and coding tasks. Claude 3 Opus achieves state-of-the-art results on several evaluations, while Haiku performs comparably to Claude 2 on text-based tasks, and Sonnet and Opus significantly surpass it. Moreover, these models exhibit enhanced fluency in non-English languages, enhancing their versatility for a global audience. The report also includes an in-depth analysis of evaluations, focusing on core capabilities, safety considerations, societal impacts, and adherence to Responsible Scaling Policy. | Foundational LLM|

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
14. [LLM Foundation Models from the Ground Up | Primer](https://www.youtube.com/watch?v=W0c7jQezTDw&list=PLTPXxbhUt-YWjMCDahwdVye8HW69p5NYS) by Databricks
15. [Generative AI Explained](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-07+V1/) by Nvidia
16. [Transformer Models and BERT Model](https://www.cloudskillsboost.google/course_templates/538) by Google Cloud
17. [Generative AI Learning Plan for Decision Makers](https://explore.skillbuilder.aws/learn/public/learning_plan/view/1909/generative-ai-learning-plan-for-decision-makers) by AWS
18. [Introduction to Responsible AI](https://www.cloudskillsboost.google/course_templates/554) by Google Cloud
19. [Fundamentals of Generative AI](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/) by Microsoft Azure
20. [Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-122979-leestott)  by Microsoft
21. [ChatGPT for Beginners: The Ultimate Use Cases for Everyone](https://www.udemy.com/course/chatgpt-for-beginners-the-ultimate-use-cases-for-everyone/) by Udemy
22. [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) by Andrej Karpathy
23. [ChatGPT for Everyone](https://learnprompting.org/courses/chatgpt-for-everyone) by Learn Prompting
    



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
13. [Building RAG Agents with LLMs](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-15+V1/) by Nvidia
14. [Building Generative AI Applications Using Amazon Bedrock](https://explore.skillbuilder.aws/learn/course/external/view/elearning/17904/building-generative-ai-applications-using-amazon-bedrock-aws-digital-training) by AWS
15. [Efficiently Serving LLMs](https://learn.deeplearning.ai/courses/efficiently-serving-llms/lesson/1/introduction) by DeepLearning.AI
16. [Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) by DeepLearning.AI
17. [Serverless LLM apps with Amazon Bedrock](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/) by DeepLearning.AI
18. [Building Applications with Vector Databases](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/) by DeepLearning.AI
19. [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) by DeepLearning.AI
20. [LLMOps](https://www.deeplearning.ai/short-courses/llmops/) by DeepLearning.AI
21. [Build LLM Apps with LangChain.js](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/) by DeepLearning.AI
22. [Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) by DeepLearning.AI
23. [Operationalizing LLMs on Azure](https://www.coursera.org/learn/llmops-azure) by Coursera
24. [Generative AI Full Course – Gemini Pro, OpenAI, Llama, Langchain, Pinecone, Vector Databases & More](https://www.youtube.com/watch?v=mEsleV16qdo) by freeCodeCamp.org
25. [Training & Fine-Tuning LLMs for Production](https://learn.activeloop.ai/courses/llms) by Activeloop
    

##### Prompt Engineering, RAG and Fine-Tuning
1. [LangChain & Vector Databases in Production](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVhnQW8xNDdhSU9IUDVLXzFhV2N0UkNRMkZrQXxBQ3Jtc0traUxHMzZJcGJQYjlyckYxaGxYVWlsOFNGUFlFVEdhNzdjTWpPUlQ2TF9XczRqNkxMVGpJTnd5YmYzV0prQ0IwZURNcHhIZ3h1Z051VTl5MXBBLUN0dkM0NHRkQTFua1Jpc0VCRFJUb0ZQZG95b0JqMA&q=https%3A%2F%2Flearn.activeloop.ai%2Fcourses%2Flangchain&v=gKUTDC13jys) by Activeloop

2. [Reinforcement Learning from Human Feedback](https://learn.deeplearning.ai/reinforcement-learning-from-human-feedback) by DeepLearning.AI

3. [Building Applications with Vector Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by DeepLearning.AI

4. [Finetuning Large Language Models](https://learn.deeplearning.ai/finetuning-large-language-models) by Deeplearning.AI
5. [LangChain: Chat with Your Data](http://learn.deeplearning.ai/langchain-chat-with-your-data/) by Deeplearning.AI

6. [Building Systems with the ChatGPT API](https://learn.deeplearning.ai/chatgpt-building-system) by Deeplearning.AI
7. [Prompt Engineering with Llama 2](https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/) by Deeplearning.AI
8. [Building Applications with Vector Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by Deeplearning.AI
9. [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction) by Deeplearning.AI
10. [Advanced RAG Orchestration series](https://www.youtube.com/watch?v=CeDS1yvw9E4) by LlamaIndex
11. [Prompt Engineering Specialization](https://www.coursera.org/specializations/prompt-engineering) by Coursera
12. [Augment your LLM Using Retrieval Augmented Generation](https://courses.nvidia.com/courses/course-v1:NVIDIA+S-FX-16+v1/) by Nvidia
13. [Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/)  by Deeplearning.AI
14. [Open Source Models with Hugging Face](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/) by Deeplearning.AI
15. [Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) by Deeplearning.AI
16. [Understanding and Applying Text Embeddings](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/) by Deeplearning.AI
17. [JavaScript RAG Web Apps with LlamaIndex](https://www.deeplearning.ai/short-courses/javascript-rag-web-apps-with-llamaindex/) by Deeplearning.AI
18. [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/) by Deeplearning.AI
19. [Preprocessing Unstructured Data for LLM Applications](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/) by Deeplearning.AI
20. [Retrieval Augmented Generation for Production with LangChain & LlamaIndex](https://learn.activeloop.ai/courses/rag)  by Activeloop
21. [Introduction to Prompt Engineering](https://learnprompting.org/courses/intro-to-prompt-engineering) by Learn Prompting
22. [Advanced Prompt Engineering](https://learnprompting.org/courses/advanced-prompt-engineering) by Learn Prompting
23. [Prompt Engineering for Marketing](https://learnprompting.org/courses/prompt-engineering-for-marketing) by Learn Prompting
24. [Retrieval-Augmented Generation](https://learnprompting.org/courses/rag) by Learn Prompting
    

##### Evaluation
1. [Building and Evaluating Advanced RAG Applications](https://learn.deeplearning.ai/building-evaluating-advanced-rag) by DeepLearning.AI
2. [Evaluating and Debugging Generative AI Models Using Weights and Biases](https://learn.deeplearning.ai/evaluating-debugging-generative-ai) by Deeplearning.AI
3. [Quality and Safety for LLM Applications](https://www.deeplearning.ai/short-courses/quality-safety-llm-applications/) by Deeplearning.AI
4. [Red Teaming LLM Applications](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/?utm_campaign=giskard-launch&utm_medium=headband&utm_source=dlai-homepage) by Deeplearning.AI
   

##### Multimodal 
1. [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) by DeepLearning.AI
2. [How to Use Midjourney, AI Art and ChatGPT to Create an Amazing Website](https://www.youtube.com/watch?v=5wdCev86RYE) by Brad Hussey
3. [Build AI Apps with ChatGPT, DALL-E and GPT-4](https://scrimba.com/learn/buildaiapps) by Scrimba
4. [11-777: Multimodal Machine Learning](https://www.youtube.com/playlist?list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA) by Carnegie Mellon University
5. [Image Creation with DALL·E 3](https://learnprompting.org/courses/image-creation-with-dalle3) by Learn Prompting
6. [Introduction to Runway](https://learnprompting.org/courses/runway) by Learn Prompting

#### Miscellaneous 
1. [Avoiding AI Harm](https://www.coursera.org/learn/avoiding-ai-harm) by Coursera
2. [Developing AI Policy](https://www.coursera.org/learn/developing-ai-policy) by Coursera
3. [AI Safety](https://learnprompting.org/courses/ai-safety) by Learn Prompting
   

   
   



   



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


