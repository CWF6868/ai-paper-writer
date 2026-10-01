# corpus.py —— RAG 学术文献语料库（唯一数据源）
# 每条文献：id 唯一、text 用于向量化（中文摘要+关键词，便于中文查询命中）、
# metadata 为标题/作者/出处/年份（真实、可查证的经典论文）。
# 灌库统一走 seed_rag.py，避免在其它地方重复维护这份数据。

ACADEMIC_REFERENCES = [
    # ---- 大语言模型 / 基础模型 ----
    {
        "id": "ref-001",
        "text": "Transformer 架构论文，提出完全基于自注意力机制的序列建模方法，是后续 GPT、BERT 等所有大语言模型的基石。关键词：注意力机制、transformer、序列建模、自注意力。",
        "metadata": {"title": "Attention Is All You Need", "authors": "Vaswani 等", "journal": "NeurIPS", "year": 2017},
    },
    {
        "id": "ref-002",
        "text": "BERT 预训练语言模型，提出双向 Transformer 编码器与掩码语言模型、下一句预测两种预训练任务，显著提升多项自然语言理解任务效果。关键词：预训练、BERT、掩码语言模型、自然语言理解。",
        "metadata": {"title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "authors": "Devlin 等", "journal": "NAACL", "year": 2019},
    },
    {
        "id": "ref-003",
        "text": "GPT-2 语言模型，展示大规模无监督语料在多任务上的泛化能力，提出语言模型是无监督多任务学习者的观点。关键词：GPT、语言模型、无监督学习、文本生成。",
        "metadata": {"title": "Language Models are Unsupervised Multitask Learners", "authors": "Radford 等", "journal": "OpenAI", "year": 2019},
    },
    {
        "id": "ref-004",
        "text": "GPT-3 千亿参数语言模型，证明在极少示例甚至零示例提示下即可完成翻译、问答、写作等任务，提出上下文学习范式。关键词：GPT-3、少样本学习、上下文学习、大模型。",
        "metadata": {"title": "Language Models are Few-Shot Learners", "authors": "Brown 等", "journal": "NeurIPS", "year": 2020},
    },
    {
        "id": "ref-005",
        "text": "InstructGPT 指令微调方法，利用人类反馈强化学习让语言模型更准确地遵循用户指令，是 ChatGPT 的重要基础。关键词：指令微调、人类反馈、强化学习、对齐。",
        "metadata": {"title": "Training language models to follow instructions with human feedback", "authors": "Ouyang 等", "journal": "NeurIPS", "year": 2022},
    },
    {
        "id": "ref-006",
        "text": "LLaMA 开源基础语言模型，探索在更多数据上训练更小模型以获得更好性能，推动开源大模型生态发展。关键词：LLaMA、开源模型、基础模型、训练效率。",
        "metadata": {"title": "LLaMA: Open and Efficient Foundation Language Models", "authors": "Touvron 等", "journal": "arXiv", "year": 2023},
    },
    {
        "id": "ref-007",
        "text": "Chinchilla 缩放定律论文，系统研究模型参数量与训练数据量的最优配比，提出在给定算力下模型大小与数据量应同比放大的结论。关键词：缩放定律、Chinchilla、算力、训练数据。",
        "metadata": {"title": "Training Compute-Optimal Large Language Models", "authors": "Hoffmann 等", "journal": "NeurIPS", "year": 2022},
    },
    {
        "id": "ref-008",
        "text": "GPT-4 技术报告，介绍多模态大模型的训练、评测与安全性，展示在多项专业考试和任务上达到人类水平的表现。关键词：GPT-4、多模态、评测、安全。",
        "metadata": {"title": "GPT-4 Technical Report", "authors": "OpenAI", "journal": "arXiv", "year": 2023},
    },
    {
        "id": "ref-009",
        "text": "Llama 2 开源对话模型，发布预训练与人类反馈微调的对话版本，并开源模型权重供研究与商用。关键词：Llama 2、对话模型、开源、指令微调。",
        "metadata": {"title": "Llama 2: Open Foundation and Fine-Tuned Chat Models", "authors": "Touvron 等", "journal": "arXiv", "year": 2023},
    },
    # ---- 检索增强生成 / 推理 ----
    {
        "id": "ref-010",
        "text": "RAG 检索增强生成方法，将外部知识检索与生成模型结合，有效缓解大模型幻觉问题，被广泛应用于问答与知识密集型任务。关键词：检索增强生成、RAG、知识检索、幻觉。",
        "metadata": {"title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "authors": "Lewis 等", "journal": "NeurIPS", "year": 2020},
    },
    {
        "id": "ref-011",
        "text": "思维链提示方法，通过在提示中引导模型逐步推理，显著提升大语言模型在数学、逻辑等复杂推理任务上的表现。关键词：思维链、提示工程、推理、大模型。",
        "metadata": {"title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "authors": "Wei 等", "journal": "NeurIPS", "year": 2022},
    },
    {
        "id": "ref-012",
        "text": "ReAct 框架将推理与行动结合，让语言模型交替进行思考和调用外部工具，提升决策与问答能力。关键词：ReAct、推理与行动、智能体、工具调用。",
        "metadata": {"title": "ReAct: Synergizing Reasoning and Acting in Language Models", "authors": "Yao 等", "journal": "ICLR", "year": 2023},
    },
    {
        "id": "ref-013",
        "text": "思维树方法，将问题求解建模为树状搜索，允许模型探索多条推理路径并回溯，提升复杂规划任务表现。关键词：思维树、树搜索、规划、推理。",
        "metadata": {"title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "authors": "Yao 等", "journal": "NeurIPS", "year": 2023},
    },
    {
        "id": "ref-014",
        "text": "自洽性方法，通过对同一问题采样多条推理链并投票，显著提升思维链推理的稳定性与准确率。关键词：自洽性、多路径推理、投票、推理增强。",
        "metadata": {"title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "authors": "Wang 等", "journal": "ICLR", "year": 2023},
    },
    {
        "id": "ref-015",
        "text": "稠密段落检索 DPR，采用双编码器结构学习问题和段落的稠密向量表示，是开放域问答检索的主流方法。关键词：稠密检索、DPR、开放域问答、双编码器。",
        "metadata": {"title": "Dense Passage Retrieval for Open-Domain Question Answering", "authors": "Karpukhin 等", "journal": "EMNLP", "year": 2020},
    },
    {
        "id": "ref-016",
        "text": "LoRA 低秩适应方法，通过冻结预训练权重并注入可训练的低秩矩阵，实现大模型的高效参数微调。关键词：LoRA、低秩适应、参数高效微调、大模型。",
        "metadata": {"title": "LoRA: Low-Rank Adaptation of Large Language Models", "authors": "Hu 等", "journal": "ICLR", "year": 2022},
    },
    {
        "id": "ref-017",
        "text": "FlashAttention 注意力加速算法，通过分块和显存感知优化，将注意力计算的时间和显存开销大幅降低，成为大模型训练标准组件。关键词：FlashAttention、注意力加速、显存优化、训练效率。",
        "metadata": {"title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "authors": "Dao 等", "journal": "NeurIPS", "year": 2022},
    },
    # ---- 自然语言处理基础 ----
    {
        "id": "ref-018",
        "text": "word2vec 词向量方法，提出跳字和连续词袋模型，用神经网络学习词的分布式表示，是词嵌入的经典基础工作。关键词：词向量、word2vec、分布式表示、词嵌入。",
        "metadata": {"title": "Distributed Representations of Words and Phrases and their Compositionality", "authors": "Mikolov 等", "journal": "NeurIPS", "year": 2013},
    },
    {
        "id": "ref-019",
        "text": "Seq2Seq 序列到序列学习，用编码器-解码器结构实现序列转换，是机器翻译等生成任务的奠基性工作。关键词：序列到序列、编码器解码器、机器翻译、文本生成。",
        "metadata": {"title": "Sequence to Sequence Learning with Neural Networks", "authors": "Sutskever 等", "journal": "NeurIPS", "year": 2014},
    },
    {
        "id": "ref-020",
        "text": "注意力机制的机器翻译，提出在解码时动态对齐源端信息，解决长句翻译的信息瓶颈问题，是注意力机制的早期经典应用。关键词：注意力、机器翻译、对齐、编码器解码器。",
        "metadata": {"title": "Neural Machine Translation by Jointly Learning to Align and Translate", "authors": "Bahdanau 等", "journal": "ICLR", "year": 2015},
    },
    {
        "id": "ref-021",
        "text": "GloVe 全局词向量方法，利用全局共现统计与局部上下文结合训练词嵌入，是词向量表示的代表方法之一。关键词：GloVe、全局共现、词嵌入、语义表示。",
        "metadata": {"title": "GloVe: Global Vectors for Word Representation", "authors": "Pennington 等", "journal": "EMNLP", "year": 2014},
    },
    {
        "id": "ref-022",
        "text": "T5 统一文本到文本格式的预训练模型，将各类自然语言处理任务统一为文本生成问题，系统探索迁移学习的极限。关键词：T5、文本到文本、预训练、迁移学习。",
        "metadata": {"title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "authors": "Raffel 等", "journal": "JMLR", "year": 2020},
    },
    # ---- 计算机视觉 / 多模态 ----
    {
        "id": "ref-023",
        "text": "AlexNet 深度卷积神经网络，使用 ReLU、Dropout 和大规模数据训练，在 ImageNet 竞赛中以巨大优势夺冠，开启深度学习时代。关键词：AlexNet、卷积神经网络、图像分类、深度学习。",
        "metadata": {"title": "ImageNet Classification with Deep Convolutional Neural Networks", "authors": "Krizhevsky 等", "journal": "NeurIPS", "year": 2012},
    },
    {
        "id": "ref-024",
        "text": "VGG 卷积网络，通过堆叠小卷积核加深网络深度，验证了网络深度对图像识别性能的重要性。关键词：VGG、卷积网络、网络深度、图像识别。",
        "metadata": {"title": "Very Deep Convolutional Networks for Large-Scale Image Recognition", "authors": "Simonyan 等", "journal": "ICLR", "year": 2015},
    },
    {
        "id": "ref-025",
        "text": "ResNet 残差网络，引入跳跃连接解决深层网络训练退化问题，使训练上百上千层网络成为可能。关键词：ResNet、残差连接、深层网络、梯度消失。",
        "metadata": {"title": "Deep Residual Learning for Image Recognition", "authors": "He 等", "journal": "CVPR", "year": 2016},
    },
    {
        "id": "ref-026",
        "text": "生成对抗网络 GAN，通过生成器与判别器的对抗训练生成逼真数据，是生成模型的重要里程碑。关键词：生成对抗网络、GAN、生成器、对抗训练。",
        "metadata": {"title": "Generative Adversarial Networks", "authors": "Goodfellow 等", "journal": "NeurIPS", "year": 2014},
    },
    {
        "id": "ref-027",
        "text": "YOLO 目标检测算法，将检测建模为单次回归问题，实现实时端到端的目标检测，速度远超两阶段方法。关键词：YOLO、目标检测、实时检测、单阶段检测。",
        "metadata": {"title": "You Only Look Once: Unified, Real-Time Object Detection", "authors": "Redmon 等", "journal": "CVPR", "year": 2016},
    },
    {
        "id": "ref-028",
        "text": "U-Net 编解码分割网络，通过跳跃连接融合多尺度特征，在小样本医学图像分割上表现优异。关键词：U-Net、图像分割、医学图像、编解码结构。",
        "metadata": {"title": "U-Net: Convolutional Networks for Biomedical Image Segmentation", "authors": "Ronneberger 等", "journal": "MICCAI", "year": 2015},
    },
    {
        "id": "ref-029",
        "text": "CLIP 多模态模型，通过图文对比学习统一视觉与文本表示，实现零样本图像分类，是视觉语言模型的重要基础。关键词：CLIP、对比学习、多模态、图文对齐。",
        "metadata": {"title": "Learning Transferable Visual Models From Natural Language Supervision", "authors": "Radford 等", "journal": "ICML", "year": 2021},
    },
    {
        "id": "ref-030",
        "text": "潜在扩散模型（Stable Diffusion），在压缩潜在空间中进行扩散建模，大幅降低文生图的计算开销，推动图像生成普及。关键词：潜在扩散、Stable Diffusion、文生图、扩散模型。",
        "metadata": {"title": "High-Resolution Image Synthesis with Latent Diffusion Models", "authors": "Rombach 等", "journal": "CVPR", "year": 2022},
    },
    {
        "id": "ref-031",
        "text": "DALL·E 文生图模型，离散变分自编码器与自回归 Transformer 结合，根据自然语言描述生成图像。关键词：DALL·E、文生图、文本到图像、生成模型。",
        "metadata": {"title": "Zero-Shot Text-to-Image Generation", "authors": "Ramesh 等", "journal": "ICML", "year": 2021},
    },
    {
        "id": "ref-032",
        "text": "Segment Anything 通用图像分割模型，提出可提示的分割任务与大规模数据引擎，实现强泛化的任意目标分割。关键词：图像分割、SAM、提示分割、基础模型。",
        "metadata": {"title": "Segment Anything", "authors": "Kirillov 等", "journal": "ICCV", "year": 2023},
    },
    # ---- 强化学习 ----
    {
        "id": "ref-033",
        "text": "深度 Q 网络 DQN，将卷积神经网络与 Q 学习结合，在 Atari 游戏上直接从原始像素学习控制策略。关键词：深度强化学习、DQN、Q学习、经验回放。",
        "metadata": {"title": "Playing Atari with Deep Reinforcement Learning", "authors": "Mnih 等", "journal": "arXiv", "year": 2013},
    },
    {
        "id": "ref-034",
        "text": "人类水平控制的深度强化学习，DQN 在多种 Atari 游戏上达到甚至超越人类玩家水平。关键词：深度强化学习、Atari、人类水平、DQN。",
        "metadata": {"title": "Human-level control through deep reinforcement learning", "authors": "Mnih 等", "journal": "Nature", "year": 2015},
    },
    {
        "id": "ref-035",
        "text": "AlphaGo 围棋系统，融合深度神经网络与蒙特卡洛树搜索，首次击败人类顶级围棋选手。关键词：AlphaGo、围棋、蒙特卡洛树搜索、强化学习。",
        "metadata": {"title": "Mastering the game of Go with deep neural networks and tree search", "authors": "Silver 等", "journal": "Nature", "year": 2016},
    },
    {
        "id": "ref-036",
        "text": "近端策略优化 PPO，通过截断重要性采样和策略更新稳定训练过程，是当前主流的强化学习算法之一。关键词：PPO、近端策略优化、策略梯度、强化学习。",
        "metadata": {"title": "Proximal Policy Optimization Algorithms", "authors": "Schulman 等", "journal": "arXiv", "year": 2017},
    },
    # ---- 图神经网络 / 知识图谱 / 联邦学习 / 优化 ----
    {
        "id": "ref-037",
        "text": "图卷积网络 GCN，将卷积操作推广到图结构数据，实现节点级半监督分类的端到端学习。关键词：图卷积网络、GCN、图表示学习、半监督学习。",
        "metadata": {"title": "Semi-Supervised Classification with Graph Convolutional Networks", "authors": "Kipf 等", "journal": "ICLR", "year": 2017},
    },
    {
        "id": "ref-038",
        "text": "图注意力网络 GAT，在图节点聚合中引入注意力机制，为不同邻居分配不同权重。关键词：图注意力、GAT、图神经网络、注意力机制。",
        "metadata": {"title": "Graph Attention Networks", "authors": "Veličković 等", "journal": "ICLR", "year": 2018},
    },
    {
        "id": "ref-039",
        "text": "TransE 知识图谱嵌入方法，将实体和关系映射到向量空间，用平移向量建模三元组关系，是知识图谱表示的经典基线。关键词：TransE、知识图谱、知识表示、表示学习。",
        "metadata": {"title": "Translating Embeddings for Modeling Multi-relational Data", "authors": "Bordes 等", "journal": "NeurIPS", "year": 2013},
    },
    {
        "id": "ref-040",
        "text": "Adam 优化算法，结合动量和自适应学习率，成为深度学习训练最常用的优化器。关键词：Adam、优化算法、自适应学习率、深度学习训练。",
        "metadata": {"title": "Adam: A Method for Stochastic Optimization", "authors": "Kingma 等", "journal": "ICLR", "year": 2015},
    },
    {
        "id": "ref-041",
        "text": "联邦学习，通过多客户端协作训练模型而无需共享原始数据，保护数据隐私，是分布式与隐私保护学习的重要方向。关键词：联邦学习、隐私保护、分布式学习、模型聚合。",
        "metadata": {"title": "Communication-Efficient Learning of Deep Networks from Decentralized Data", "authors": "McMahan 等", "journal": "AISTATS", "year": 2017},
    },
    {
        "id": "ref-042",
        "text": "Dropout 正则化方法，训练时随机丢弃神经单元防止过拟合，显著提升深度网络的泛化能力。关键词：Dropout、正则化、过拟合、深度学习。",
        "metadata": {"title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "authors": "Srivastava 等", "journal": "JMLR", "year": 2014},
    },
]