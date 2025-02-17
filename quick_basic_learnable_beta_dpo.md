**项目名称：**  基于 Learnable Beta DPO 的 Qwen1.5B 人类偏好对齐微调项目

**项目目标：**

开发一个完整的、可复现的微调流程，使用 Learnable Beta DPO (Direct Preference Optimization) 算法，结合可学习的 `BetaHead` 网络结构，对 Qwen1.5B 大语言模型进行微调，使其在特定领域或通用场景下更好地对齐人类偏好，提升生成内容的质量和符合人类期望的程度。

**核心创新点：**

本项目采用 **Learnable Beta DPO** 方法，核心在于使用与策略模型紧密耦合的 `BetaHead` 子网络来动态调整 DPO 算法中的 $\beta$ 参数。`BetaHead` 网络直接利用策略模型的内部表征（最后一层隐状态），结合策略模型计算的困惑度 (PPL)，通过一个轻量级的可学习网络 $f(h_{\pi_\theta}(x))$ 实现对探索-利用平衡的精细控制。这种共享表征的设计不仅提高了计算效率，更重要的是实现了策略模型和 BetaHead 的协同进化，有望在 DPO 微调过程中获得更优的性能和更稳定的训练。

**预期成果：**

1.  **可运行的代码库:**  提供完整、模块化、注释清晰的 Python 代码库，包含数据处理、模型构建、训练脚本、评估脚本等，确保代码可读性、可维护性和可扩展性。
2.  **Learnable Beta DPO 微调流程:**  实现基于 `BetaHead` 网络的 Learnable Beta DPO 微调流程，能够加载 Qwen1.5B 模型，使用偏好数据集进行微调，并保存微调后的模型。
3.  **动态 Beta 输出功能:**  成功集成 `BetaHead` 网络，实现根据输入上下文动态输出 $\beta(x)$ 值的功能，并在 DPO Loss 计算中使用动态 $\beta$。
4.  **模型评估报告:**  提供详细的实验报告，包括超参数设置、训练日志、评估指标（包括但不限于 Loss 曲线、人工评估、自动评估指标等），以及 Learnable Beta DPO 相对于 Fixed Beta DPO 的性能对比分析（如果条件允许）。
5.  **微调后的 Qwen1.5B 模型:**  提供微调后的 Qwen1.5B 模型权重，可以直接加载使用。

**详细开发任务分解：**

**1. 环境搭建与依赖安装:**

*   **任务:**  创建 Python 虚拟环境，安装所有必要的依赖库，包括：
    *   `transformers` (最新稳定版，需兼容 Qwen1.5B)
    *   `torch` (与 CUDA 版本匹配)
    *   `datasets`
    *   `accelerate` (可选，用于分布式训练)
    *   `sentencepiece` (Qwen tokenizer 依赖)
    *   `tensorboard` 或 `wandb` (可选，用于训练监控)
*   **交付标准:**  提供 `requirements.txt` 文件，包含所有依赖库及其版本。确保环境配置文档清晰，可复现。

**2.  `BetaHead` 网络模块开发:**

*   **任务:**  根据已有的设计方案，在 `beta_head.py` 文件中实现 `BetaHead` 和 `DynamicBetaDPOModel` 类。
    *   **`BetaHead` 类:**
        *   实现 `__init__` 方法，包括可学习参数 `w` 的初始化，神经网络 `NN(x)` 的构建 (至少实现线性层版本，并预留 MLP 和 Transformer 结构的扩展接口)，以及 `epsilon` 参数的设置。
        *   实现 `forward(context_embedding, ppl)` 方法，根据公式 $\beta(x) = w \cdot PPL(x) \cdot f(x)$ 计算并返回动态 $\beta$ 值。确保代码注释清晰，参数可配置。
        * context_embedding 提取策略: 本项目默认采用取最后一个 token 的 last hidden state 的方法来提取 context_embedding。这种方法在理论上更符合自回归 Decoder-only 模型的特性，且实践中已被证明有效。当然，在后续实验中，也可以探索其他 context_embedding 提取策略 (例如平均池化) 并进行比较。
    *   **`DynamicBetaDPOModel` 类:**
        *   实现 `__init__(awesome_lm, beta_head)` 方法，接收 `AwesomeLanguageModel` (此处为 Qwen1.5B 模型实例) 和 `BetaHead` 实例作为输入。
        *   实现 `get_dynamic_beta(context_token_ids, attention_mask)` 方法，负责调用 `AwesomeLanguageModel` 计算 PPL 和 context embedding，并将结果传递给 `BetaHead` 计算动态 $\beta$。
        *   实现 `forward_lm(context_token_ids, attention_mask)` 方法，直接调用 `AwesomeLanguageModel` 的 forward 函数，返回 logits。
*   **交付标准:**  提供 `beta_head.py` 文件，包含完整、可运行的 `BetaHead` 和 `DynamicBetaDPOModel` 代码。代码风格符合 PEP 8 规范，注释清晰。

**3.  Qwen1.5B 模型集成:**

*   **任务:**  编写代码加载 Qwen1.5B 模型和 tokenizer，并将其集成到 `DynamicBetaDPOModel` 中。
    *   使用 `transformers` 库加载预训练的 Qwen1.5B 模型 (指定模型路径或 Hugging Face 模型名，注意 `trust_remote_code=True`)。
    *   确保 tokenizer 正确加载，并处理 Qwen tokenizer 可能需要的特殊设置 (例如 pad token)。
    *   在 `DynamicBetaDPOModel` 中正确使用 Qwen1.5B 模型进行 PPL 计算、context embedding 获取和 logits 输出。
*   **交付标准:**  提供模型加载和集成的代码示例，确保 `DynamicBetaDPOModel` 可以成功加载并使用 Qwen1.5B 模型。

**4.  偏好数据集处理:**

*   **任务:**  编写数据处理代码，加载和预处理偏好数据集。
    *   支持加载常见格式的偏好数据集 (例如 Hugging Face datasets 格式，或 CSV, JSON 等)。
    *   实现数据预处理函数，包括：
        *   文本 tokenization (使用 Qwen tokenizer)。
        *   padding 和 truncation (根据需求设定最大长度)。
        *   构建 DataLoader，用于批量加载训练数据。
*   **数据集格式要求:**  数据集应包含 "prompt", "chosen_response", "rejected_response" 字段。
*   **交付标准:**  提供数据加载和预处理的代码，以及数据 DataLoader 的创建示例。 确保数据处理流程高效、正确。

**5.  Learnable Beta DPO 训练脚本开发:**

*   **任务:**  编写 Learnable Beta DPO 的训练脚本 (`train.py`)。
    *   实现完整的训练循环，包括：
        *   数据加载 (使用 DataLoader)。
        *   动态 $\beta$ 值计算 (使用 `DynamicBetaDPOModel.get_dynamic_beta`)。
        *   logits 计算 (使用 `DynamicBetaDPOModel.forward_lm`)。
        *   Learnable Beta DPO Loss 计算 (需要实现 DPO Loss 函数，并使用动态 $\beta$)。
        *   反向传播和优化 (使用 AdamW 或其他合适的优化器)。
        *   训练日志记录 (loss 值，动态 $\beta$ 值的统计信息等)。
        *   模型 checkpoint 保存 (定期保存模型权重)。
    *   训练脚本应支持超参数配置 (例如学习率, batch size, epoch 数, BetaHead 参数等)，并可以通过命令行参数或配置文件进行设置。
*   **交付标准:**  提供可运行的 `train.py` 训练脚本。 脚本应结构清晰，逻辑正确，并包含必要的注释和日志记录。

**6.  模型评估脚本开发:**

*   **任务:**  编写模型评估脚本 (`evaluate.py`)。
    *   实现模型加载功能 (加载微调后的 Qwen1.5B 模型)。
    *   实现评估指标计算功能 (至少包括 Loss 值，并根据具体任务需求，选择合适的人工评估或自动评估指标)。
    *   提供评估报告生成功能，清晰展示评估结果。
*   **交付标准:**  提供可运行的 `evaluate.py` 评估脚本。 脚本应能够加载微调后的模型，并输出评估结果。

**7.  实验报告撰写:**

*   **任务:**  撰写详细的实验报告 (`report.md` 或 `report.pdf`)。
    *   **内容应包括:**
        *   项目概述和目标
        *   Learnable Beta DPO 方法的原理和优势
        *   模型架构和参数设置 (Qwen1.5B 版本, BetaHead 结构和超参数)
        *   数据集描述和统计信息
        *   训练过程描述 (超参数设置, 训练时长, Loss 曲线等)
        *   评估指标和评估结果 (详细展示评估指标数值，并进行结果分析)
        *   Learnable Beta DPO 的效果分析 (例如，动态 $\beta$ 值的分布和变化趋势分析，以及与 Fixed Beta DPO 的对比分析，如果进行了对比实验)
        *   结论和未来工作展望
*   **交付标准:**  提供详细、专业的实验报告文档 (`report.md` 或 `report.pdf`)，内容完整、逻辑清晰、数据翔实。

**技术要求:**

*   熟练掌握 Python 编程。
*   熟悉 PyTorch 深度学习框架。
*   熟悉 Hugging Face `transformers` 和 `datasets` 库。
*   理解 DPO (Direct Preference Optimization) 算法原理。
*   了解大语言模型 (LLM) 的微调技术。
*   具备良好的代码编写习惯和文档撰写能力。


**验收标准：**

*   所有交付成果完整、符合要求。
*   代码可运行、无明显 bug，并易于理解和维护。
*   模型微调成功，评估指标达到预期水平 (具体指标和预期水平在项目启动前进一步明确)。
*   实验报告内容完整、分析深入、结论合理。

请 AI 或开发人员严格按照以上提示词进行项目开发，确保高质量地完成基于 Learnable Beta DPO 的 Qwen1.5B 人类偏好对齐微调项目。