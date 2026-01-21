Beyond the Context Window: A Comprehensive Analysis of Recursive Language Models and Infinite Context Inference Architectures
Executive Summary
The trajectory of Large Language Model (LLM) development over the past decade has been defined by a singular, persistent architectural constraint: the finite context window. While parameter counts have scaled exponentially, the attention mechanism—the core engine of the Transformer architecture—has imposed a quadratic tax on sequence length, creating a formidable barrier to the processing of truly massive datasets. Despite recent engineering triumphs that have extended context windows from 4,096 tokens to over 10 million, a phenomenon known as "context rot" persists, where reasoning fidelity degrades as information density increases. This report posits that the solution to the infinite context problem lies not in the further scaling of passive memory architectures, but in a paradigm shift towards Recursive Language Models (RLMs).
RLMs represent a transition from "In-Context Learning" to "Inference-Time Computing." By treating context not as a static tensor input but as an external state variable within a Turing-complete environment (typically a Python REPL), RLMs transform the challenge of long-context processing from a memory bottleneck into a program synthesis task. This report provides an exhaustive technical analysis of this emerging field, synthesizing data from over 130 research artifacts. It explores the theoretical underpinnings of "Context-as-State," details the prompt architectures of leading recursive frameworks (including RLM, RecurrentGPT, MemWalker, and InftyThink), and presents empirical evidence of their logarithmic scaling capabilities.
The analysis reveals that RLMs achieve "cost inversion" for massive tasks—becoming cheaper than standard models as context grows—and solve the "Lost-in-the-Middle" phenomenon by replacing probabilistic attention with deterministic code execution. However, this agency introduces new fragility vectors, including code hallucinations and sequential latency, which require novel prompt engineering and sandboxing strategies to mitigate.
Part I: The Context Crisis and the Limits of Attention
1.1 The Quadratic Barrier and the Illusion of Capacity
Since the introduction of the Transformer architecture in 2017, the fundamental bottleneck for handling long sequences has been the self-attention mechanism. Mathematically, for a sequence of length $N$, the model must compute an $N \times N$ attention matrix, resulting in computational complexity of $O(N^2)$ and memory usage that scales similarly. While optimizations like FlashAttention, Ring Attention, and linear attention variants have successfully pushed the physical limits of the context window to the order of 10 million tokens 1, physical capacity does not equate to cognitive capacity.
Research consistently indicates a divergence between "effective context" and "allocatable context." As the context window fills, the attention distribution flattens, leading to attention dispersion. The model becomes less capable of retrieving specific, nuanced information ("needles") buried within the massive text ("haystack"), particularly when that information is located in the middle of the sequence—a phenomenon widely documented as "context rot" or the "lost-in-the-middle" effect.1 This degradation suggests that simply increasing the KV cache size is a brute-force solution that yields diminishing returns in reasoning quality.
1.2 The Economic Imperative for Inference-Time Scaling
Beyond the fidelity issues, the economic model of massive context is prohibitive. In a standard Transformer, processing a 10-million-token document requires encoding all 10 million tokens for every single query. Even with prefix caching, the computational overhead of attending to such a vast history for simple questions is inefficient.
The industry is thus witnessing a pivot toward Inference-Time Scaling. This paradigm shifts the burden of performance from pre-training (larger models) and memory architecture (larger windows) to test-time computation (loops, recursion, and search). The Recursive Language Model (RLM) is the flagship architecture of this shift. It trades the massive parallelism of the Transformer for the sequential, state-aware processing of a Turing machine. By breaking a problem down into sub-problems and processing them recursively, RLMs allow a model with a small, fixed context window (e.g., 8k tokens) to process inputs of effectively infinite length.5
1.3 Defining the Recursive Language Model (RLM)
It is crucial to define the RLM not as a specific neural network, but as a meta-architecture or inference strategy. An RLM wraps a standard LLM (the "Root LM") in a scaffold that provides:
External State Persistence: A mechanism to store data outside the model's weights or immediate context window (e.g., a file system, a vector database, or a Python variable).
A Recursive Primitive: A function (e.g., llm_query or rlm.call) that allows the model to invoke a fresh instance of itself.
Programmatic Agency: The ability for the model to write code to inspect, slice, and transform the external state before processing it.
This definition encompasses several specific implementations, including the MIT RLM (Zhang et al.), RecurrentGPT (Zhou et al.), and MemWalker (Chen et al.), each of which takes a different approach to managing the state transition function.3
Part II: Theoretical Foundations of Recursive Architectures
2.1 Context-as-State: The Turing Machine Analogy
To understand the power of RLMs, one must draw an analogy to the theory of computation. A standard Transformer is akin to a Finite Automaton with a very large buffer; it processes its input in a single forward pass (or autoregressively) and has no memory scratchpad that persists outside its immediate activations.
An RLM, by coupling the LLM with a Read-Eval-Print Loop (REPL), approximates a Universal Turing Machine.
The Tape: The external context variable (e.g., a string variable context in Python) represents the infinite tape. It holds more data than the model's internal registers (context window) can hold.
The Head: The LLM's current attention focus is determined by code slicing (e.g., context[10000:14000]). The model can move this "head" arbitrarily across the tape.
The State Register: The REPL environment variables and the current prompt represent the machine's state.
The Transition Function: The LLM itself acts as the transition function, reading the current state and generating the next action (code execution).1
This theoretical decoupling allows the system to solve problems that require $O(N)$ or even super-linear state tracking without requiring the underlying neural network to maintain that state in its activations.
2.2 Algorithmic Scaling Laws
The most compelling theoretical argument for RLMs is the alteration of scaling laws for retrieval and reasoning.
2.2.1 From Linear Scanning to Logarithmic Search
In a standard long-context model, finding a specific fact requires the attention mechanism to compute affinities across all tokens. In the best-case scenario (linear attention), this is $O(N)$.
An RLM, however, can synthesize a Semantic Binary Search algorithm. Consider a task to find a specific event in a 10-million-token log file:
Level 0: The RLM divides the text into 10 large sections and queries: "Which section likely contains the event?"
Level 1: It recurses into the identified section, dividing it further.
Level 2: It repeats until the relevant chunk is small enough to read directly.
This reduces the computational complexity from $O(N)$ to $O(\log N)$, enabling logarithmic scaling behaviors that are fundamentally impossible with fixed-graph neural networks.1
2.2.2 Flat Scaling vs. Collapse
Empirical observations of RLMs show "flat scaling" performance curves. While standard models like GPT-4 or Claude 3 exhibit performance collapse as context length grows (the red zone in Figure 1 of Zhang et al., 2025), RLM performance remains relatively constant regardless of total input length. As long as the decomposed chunk size remains within the model's "comfort zone" (e.g., 4k tokens), the total length of the external context is irrelevant to the fidelity of the local processing.8
2.3 Cognitive Decoupling: System 1 vs. System 2
The RLM architecture enforces a separation of concerns that mirrors the dual-process theory of cognition (System 1 vs. System 2).
System 1 (Fast, Intuitive): The LLM's forward pass is a heuristic, pattern-matching engine. It is excellent at local text generation but prone to drift over long horizons.
System 2 (Slow, Deliberate): The RLM scaffold forces the model to plan, execute code, and reflect on outputs. The recursive loop creates a "chain of thought" that is not just a text string but a series of verified computational steps.
This "Cognitive Decoupling" means that RLMs are effectively Agentic RAG systems. In traditional RAG, retrieval is performed by a non-intelligent vector database (cosine similarity). In an RLM, retrieval is performed by the LLM itself via code (e.g., writing a regex or a logical filter). This allows for "hard" logic retrieval (e.g., "Find the third paragraph after the mention of 'revenue'") which vector databases cannot perform.9
Part III: The Recursive Language Model (RLM) Architecture
The implementation of RLMs, particularly as described in the work of Zhang, Bolcato, and the MIT/Prime Intellect teams, relies on a specific set of architectural components. This section details the specifications of the "Root LM," the REPL environment, and the recursive primitive.
3.1 The Root Language Model (Depth=0)
The entry point of an RLM system is the Root LM. A critical distinction in this architecture is that the Root LM never ingests the full context.
Input: The user query $q$, the metadata of the context (e.g., len(context), file type), and a system prompt.
Operational Awareness: The Root LM is "aware" that a variable named context exists in its environment, but it treats it as an opaque object or a handle.
Function: Its primary role is not to answer the query directly but to act as a program synthesizer. It must write a plan (code) to extract the necessary information.3
3.2 The Read-Eval-Print Loop (REPL) Environment
The environment is the persistent memory bank of the RLM.
Context as Variable: The massive dataset is loaded into the memory of the host machine (not the GPU memory of the model) and bound to a variable, usually context or self.data.
Tooling: The environment provides standard Python libraries (re, math, collections). This allows the model to perform deterministic operations—counting, sorting, filtering—that neural networks struggle with.
Isolation: To prevent context pollution, the REPL captures standard output (stdout). The Root LM only sees the result of its code execution (e.g., a specific paragraph, a count, or an error message), ensuring its context window remains lean.4
3.3 The Recursive Primitive: llm_query
The defining feature of the RLM is the ability to spawn sub-processes. This is exposed to the Root LM as a Python function, typically named llm_query or rlm_call.
Technical Specification:
The llm_query function serves as the bridge between the code environment and the semantic reasoning of the LLM. It essentially spawns a new "Leaf Node" (Depth=1) model call.

Python


def llm_query(prompt: str, context_chunk: str) -> str:
    """
    Spawns a new, isolated LLM instance to process a specific chunk 
    of context with a specific instruction.
    
    Args:
        prompt: The instruction (e.g., "Summarize this section").
        context_chunk: The specific slice of text to process.
        
    Returns:
        String output from the sub-model.
    """
    #... implementation details involving API calls...


When the Root LM executes this function, the system:
Instantiates a new API call to an LLM (potentially a smaller, cheaper model like GPT-4o-mini).
Passes the restricted context_chunk and the specific prompt.
Returns the text result to the Root LM's environment variables.
Standardizes the output, allowing the Root LM to aggregate results from multiple recursive calls.1
3.4 Prompt Engineering for Recursive Architectures
The "Prompt Architecture" for RLMs is a specialized form of system design. It must explicitly ground the model in the REPL environment and define the "rules of engagement" for recursion.
System Prompt Template 4:
Role: You are an intelligent agent operating in a Recursive Language Model (RLM) environment.
Environment:
The user's input text is stored in the variable context.
The variable context contains {N} characters.
WARNING: You cannot see the full text directly. Attempting to print the full context will crash the system.
Tools:
You can write Python code to inspect, slice, and filter context.
You have access to a special function: llm_query(prompt, chunk). Use this to ask semantic questions about specific slices of the text.
Protocol:
Inspect: First, write code to check the length and structure of the data (e.g., print(context[:1000])).
Plan: Devise a strategy (e.g., Map-Reduce, Binary Search) to answer the user's query.
Execute: Write Python code to implement your plan, utilizing llm_query for sub-tasks.
Synthesize: Aggregate the results in a variable and output the final answer using FINAL(answer).
This prompt forces the model to adopt an "active explorer" persona rather than a "passive reader" persona.
3.5 Execution Trace Analysis
To illustrate the mechanism, consider a request: "Identify the primary cause of failure in this 5-million-token server log."
RLM Execution Log:
Root (Depth 0): print(len(context)) $\rightarrow$ Output: 5,240,000.
Root (Depth 0): print(context[-2000:]) (Peeking at the end for recent errors) $\rightarrow$ Output: "...System critical error: Database Lock..."
Root (Depth 0): Synthesizes a search strategy. "I need to find where the database lock first occurred."
Root (Depth 0): indices = $\rightarrow$ Output: [4800200, 4800500,...]
Root (Depth 0): Selects the first occurrence window. window = context[4800000:4800500]
Root (Depth 0): analysis = llm_query("Analyze the logs just before the database lock.", window)
Leaf (Depth 1): Processes the window. $\rightarrow$ Returns: "The logs show a slow query on the 'Orders' table initiated by user Admin."
Root (Depth 0): FINAL("The failure was caused by a database lock triggered by a slow query on the Orders table.").5
Part IV: Alternative Recursive Architectures
While the "RLM" discussed above focuses on dynamic code execution, the landscape of infinite context includes several other distinct architectures that leverage recursion and state management.
4.1 RecurrentGPT: The RNN Simulacrum
RecurrentGPT takes inspiration directly from Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks (Hochreiter & Schmidhuber, 1997). However, instead of using vector hidden states, it uses natural language paragraphs as its state.
Mechanism: It maintains a "Short-Term Memory" (a textual summary of recent generation) and a "Long-Term Memory" (stored in a VectorDB on the hard drive).
The Loop: At each timestep $t$, the model generates a paragraph of text and a "plan" for the next paragraph. It explicitly updates the short-term memory prompt and saves the generated paragraph to the VectorDB.
Prompt Architecture: The prompt simulates the LSTM cell equations. It receives {short_memory}, {current_plan}, and {retrieved_long_term_memory} as inputs and is instructed to output {new_content}, {new_short_memory}, and {new_plan}.
Distinction: RecurrentGPT is primarily designed for unbounded generation (e.g., writing a never-ending novel), whereas RLM is designed for unbounded analysis (reading a massive file). RecurrentGPT uses a fixed recurrence graph, whereas RLM uses dynamic code execution.7
4.2 MemWalker: The Tree Navigator
MemWalker addresses the infinite context problem by restructuring the data into a hierarchical tree before query time.
Construction Phase (Indexing): The long text is segmented into leaf nodes. These are summarized into parent nodes, which are summarized into higher-level nodes, creating a pyramid structure culminating in a Root Summary Node.
Navigation Phase (Inference): Upon receiving a query, the LLM starts at the Root Summary Node. It evaluates the child nodes to determine which "branch" is most likely to contain the answer. It then "walks" down the tree, recursively querying the chosen path until it reaches a leaf node with the granular data.
Comparison: This represents an $O(\log N)$ retrieval method similar to RLM's binary search. However, the structure is static (the tree is pre-computed) rather than dynamic (runtime slicing). MemWalker is highly efficient for repeated queries on the same static dataset (e.g., a legal corpus).2
4.3 InftyThink: Recursive Reasoning
InftyThink (and similar "Thought-Loop" architectures) applies recursion to the reasoning process rather than the context retrieval.
Sawtooth Memory Pattern: Instead of a monolithic Chain-of-Thought, InftyThink interleaves short reasoning segments with concise progress summaries.
Mechanism: The model generates a partial reasoning chain, summarizes its current thinking state, and passes only that summary to the next iteration. This creates a "sawtooth" pattern of memory usage (Reason -> Compress -> Reason -> Compress), allowing for effectively infinite reasoning depth without exhausting the context window.16
4.4 Prime Intellect & "Context Folding"
Prime Intellect has introduced the concept of "Context Folding" in their implementation of RLMs. This is a technique to manage the state of the agent itself over long trajectories.
The Problem: In a long running agent loop (e.g., coding a complex feature), the history of actions grows indefinitely.
The Solution: Rather than keeping the full history, the RLM recursively "folds" the completed steps into a compressed state representation. The context variable is not just text, but a dynamic object that the model reshapes.
Matryoshka State: Similar to the "Matryoshka" concept in their tooling, results from queries are bound to variables (e.g., RESULTS) and stored in the server memory, while only the metadata (e.g., "Found 150 results") enters the LLM context. This keeps the prompt lean while maintaining access to massive data.18
Part V: Empirical Analysis and Benchmarks
The theoretical promises of RLMs are validated by strong empirical data, particularly on benchmarks designed to break standard long-context models.
5.1 Needle-in-a-Haystack (NIAH) Performance
On the standard NIAH benchmark (finding a hidden fact in a large text), RLMs exhibit superior performance at extreme lengths (1M+ tokens).
Standard Models: Frontier models often fail at 100% retrieval when the needle is placed in the middle of a 1M token context (performance drops to <50% accuracy).
RLMs: Because RLMs use code-based search (Regex/String matching) or divide-and-conquer, their accuracy is independent of the needle's position. If the search logic is correct, retrieval is deterministic. RLMs consistently score >95% accuracy on NIAH tasks across 10M token inputs.5
5.2 OOLONG and Complex Reasoning
The OOLONG benchmark tests reasoning that requires aggregating information from multiple parts of the text (e.g., "List all dates mentioned in Chapter 1 and Chapter 10").
Results: RLM(GPT-5-mini) outperformed the base GPT-5 model by 114% on challenging long-context benchmarks.
Why: Standard models struggle to hold two distant pieces of information in active attention simultaneously for comparison. An RLM, using a Map-Reduce strategy, explicitly extracts both pieces into variables date_list_1 and date_list_2 and then compares them in a small context window, eliminating the distance penalty.5
5.3 The Cost Inversion Paradox
A counter-intuitive but critical finding is "Cost Inversion."
Scenario: Processing a 10M token document.
Standard LLM: A single query requires processing 10M tokens. At $2.50/1M tokens, a single query costs $25.00.
RLM: The RLM inspects the file size (0 tokens), runs a regex search (0 tokens), identifies a 50k token relevant chunk, and processes it. Total tokens processed: ~50k. Cost: $0.12.
Comparison: For massive contexts, RLMs are not just better; they are orders of magnitude cheaper. On the BrowseComp-Plus benchmark, RLM(GPT-5) had an average cost of $0.99 compared to $2.75 for the base model, while simultaneously achieving higher accuracy.4
5.4 Latency: The Achilles' Heel
The trade-off for these advantages is latency.
Sequential Bottleneck: The REPL is sequential. An RLM must write code, wait for execution, read output, and think again.
Impact: While a Transformer might process 1M tokens in parallel in 10 seconds (on a massive cluster), an RLM performing a sequential binary search might take 30-60 seconds due to the "Reason-Act-Read" loop.
Mitigation: Implementation of llm_batch allows the RLM to spawn parallel sub-queries (e.g., "Summarize these 100 chunks simultaneously"), helping to regain some parallelism.18
Part VI: Comparative Architectural Summary
The following table summarizes the key distinctions between the discussed architectures:
Feature
Standard Long-Context LLM
RAG (Retrieval Augmented Generation)
Recursive Language Model (RLM)
RecurrentGPT
MemWalker
Context Handling
Passive; Full context in KV Cache.
Retrieval via Vector Similarity.
Active; Context as External Variable.
Active; Context as Textual Memory Stream.
Pre-computed Hierarchical Tree.
Retrieval Logic
Attention Mechanism (Softmax).
Cosine Similarity.
Programmatic (Code, Regex, Logic).
Recurrent Prompting (RNN Simulation).
Interactive Tree Traversal.
Comp. Complexity
$O(N^2)$ or $O(N)$ (Linear Attn).
$O(1)$ (Approx. Nearest Neighbor).
$O(\log N)$ (Search) or $O(N)$ (Map-Reduce).
$O(N)$ (Linear Generation).
$O(\log N)$ (Tree Depth).
Primary Use Case
General Purpose (up to ~200k).
Knowledge Base Q&A.
Deep Research, Massive Data Analysis.
Infinite Creative Writing / Narrative.
Static Document Q&A.
Failure Mode
Hallucination, Attention Dispersion.
Semantic Mismatch (Lexical Gap).
Code Errors, Infinite Loops, Latency.
Memory Drift, Loss of Detail.
Tree Construction Overhead.

Part VII: Implementation Guide and Code Analysis
Implementing an RLM requires solving several engineering challenges, primarily around the security and stability of the REPL.
7.1 The "Fragile Code" Bottleneck
The reliability of an RLM is upper-bounded by the Root LM's coding ability. A model that cannot write correct regex or handles Python list indices poorly will fail.
Self-Correction: Advanced RLM implementations feed stderr (error messages) back into the context. If the model writes context and gets an "IndexError," it sees this and corrects its next attempt.
Model Selection: Code-optimized models (like Claude 3.5 Sonnet, Qwen-Coder, or DeepSeek-Coder) vastly outperform generalist models as the "Root LM" because RLM is fundamentally a coding task.4
7.2 Sandboxing and Security
Allowing an LLM to execute exec() on arbitrary code is a massive security risk.
Isolation: Implementations must use strict sandboxing (e.g., Docker, gVisor, or WASM).
Network restrictions: The REPL should generally not have internet access unless "Web Search" is a specific tool, to prevent data exfiltration.
State Management: The "context" variable must be injected into the sandbox in a read-only mode if possible, or managed carefully to prevent the model from accidentally overwriting the source data.18
7.3 Python Code Example: The Recursive Mechanism
Below is a synthesized example of how the recursive logic is implemented in the Python backend.

Python


class RecursiveContext:
    def __init__(self, large_text_content, model_client):
        self.context = large_text_content
        self.client = model_client
        self.max_depth = 2

    def llm_query(self, instruction, chunk, depth=0):
        """
        The recursive primitive exposed to the LLM.
        """
        if depth > self.max_depth:
            return "ERROR: Max recursion depth reached."
            
        # 1. Construct the prompt for the sub-agent
        system_prompt = f"""
        You are a sub-agent analysis bot. 
        Analyze the provided text chunk based on the instruction.
        Be concise and factual.
        """
        
        # 2. Call the LLM (Leaf Node)
        # Note: We pass the chunk explicitly here.
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", # Use cheaper model for leaves
            messages=
        )
        return response.choices.message.content

    def execute_root_step(self, user_query):
        """
        The Root LM loop (REPL).
        """
        # Inject the context size, but NOT the content, into the prompt
        system_prompt = f"""
        You are the Root Agent. The variable 'context' has length {len(self.context)}.
        You cannot read it all. Write Python code to inspect it.
        Use `llm_query(instruction, slice)` to analyze specific parts.
        """
        #... (Execution loop logic)...


This code demonstrates the "Context Folding" principle: the Root LM sees len(context) but not the data; the Leaf LM sees the data but not the global plan.5
Part VIII: Future Outlook and Conclusion
8.1 The Convergence of Agents and Context
RLMs blur the line between "Long Context" and "Agents." As models become better at coding and tool use, the distinction between "reading a document" and "using a tool to read a document" vanishes. We can expect future RLM iterations to move beyond simple text strings to "Context Graphs," where the model builds and navigates a knowledge graph of the data on the fly.21
8.2 Standardization via MCP
The Model Context Protocol (MCP) represents a future where RLMs can plug into any data source. Instead of just a text variable, the "Context" could be a live SQL database, a GitHub repository, or a Slack workspace. The RLM would use the same recursive query logic (llm_query) to navigate these diverse structures, effectively creating a "Universal Data Processor".19
8.3 Conclusion
Recursive Language Models represent a decisive break from the "larger context window" arms race. By reconceptualizing context as an interactive environment rather than a static input tensor, RLMs unlock $O(\log N)$ scaling properties and economic efficiencies that are physically impossible for standard Transformer architectures. They solve the "context rot" problem by replacing probabilistic attention with deterministic code execution.
While they introduce complexity in the form of code fragility and latency, they offer the only viable path to truly infinite context processing—where a model can ingest, understand, and reason over libraries of information rather than just single documents. As "System 2" reasoning becomes a priority for the next generation of AI, the RLM architecture—or a derivative thereof—is poised to become the standard interface for deep research and massive-scale data analysis.
Works cited
Recursive Language Models: Infinite Context that works - Medium, accessed January 18, 2026, https://medium.com/@pietrobolcato/recursive-language-models-infinite-context-that-works-174da45412ab
BEYOND CONTEXT LIMIT THROUGH INTERACTIVE READING, accessed January 18, 2026, https://openreview.net/pdf?id=H5XZLeXWPS
Recursive Language Models | Alex L. Zhang, accessed January 18, 2026, https://alexzhang13.github.io/blog/2025/rlm/
Recursive Language Models - arXiv, accessed January 18, 2026, https://arxiv.org/html/2512.24601v1
How Recursive Language Models Handle Infinite Input - Maxim AI, accessed January 18, 2026, https://www.getmaxim.ai/blog/breaking-the-context-window-how-recursive-language-models-handle-infinite-input/
(PDF) Recursive Language Models - ResearchGate, accessed January 18, 2026, https://www.researchgate.net/publication/399276815_Recursive_Language_Models
RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text, accessed January 18, 2026, https://openreview.net/pdf/fb7d73c7265c7132aa0436997430ec0d5662c833.pdf
Recursive Language Models - arXiv, accessed January 18, 2026, https://arxiv.org/pdf/2512.24601
(PDF) A Survey of Slow Thinking-based Reasoning LLMs using ..., accessed January 18, 2026, https://www.researchgate.net/publication/391461834_A_Survey_of_Slow_Thinking-based_Reasoning_LLMs_using_Reinforced_Learning_and_Inference-time_Scaling_Law
A Survey of Slow Thinking-based Reasoning LLMs using ... - arXiv, accessed January 18, 2026, https://arxiv.org/html/2505.02665v1
Recursive Language Models: The Complete Guide to 10M+ Token ..., accessed January 18, 2026, https://dev.to/dmitry_labintcev_9e611e04/recursive-language-models-the-future-of-10m-token-processing-and-how-to-secure-it-44h
RLM: The Ultimate Evolution of AI? Recursive Language Models, accessed January 18, 2026, https://dev.to/gaodalie_ai/rlm-the-ultimate-evolution-of-ai-recursive-language-models-3h8o
(PDF) RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text, accessed January 18, 2026, https://www.researchgate.net/publication/370949440_RecurrentGPT_Interactive_Generation_of_Arbitrarily_Long_Text
RECURRENTGPT: Interactive Generation of (Arbitrarily) Long Text, accessed January 18, 2026, https://arxiv.org/pdf/2305.13304
arXiv:2310.05029v1 [cs.CL] 8 Oct 2023, accessed January 18, 2026, https://arxiv.org/pdf/2310.05029
InftyThink: Breaking the Length Limits of Long-Context Reasoning in ..., accessed January 18, 2026, https://arxiv.org/html/2503.06692v4
Stable Sequential Test-Time Scaling for Large Reasoning Models, accessed January 18, 2026, https://arxiv.org/html/2601.09855v1
Recursive Language Models: the paradigm of 2026 - Prime Intellect, accessed January 18, 2026, https://www.primeintellect.ai/blog/rlm
How to Cut LLM Token Usage by 80% Using Recursive Document ..., accessed January 18, 2026, https://yogthos.net/posts/2026-01-16-lattice-mcp.html
alexzhang13/rlm: General plug-and-play inference library ... - GitHub, accessed January 18, 2026, https://github.com/alexzhang13/rlm
Scaling Agentic SEO through the Context Graph - Gianluca Fiorelli, accessed January 18, 2026, https://www.iloveseo.net/scaling-agentic-seo-through-the-context-graph/
