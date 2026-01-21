# RLM - Recursive Language Model Environment

A Python implementation of the **Recursive Language Model** architecture for processing documents that exceed LLM context windows.

Based on ["Beyond the Context Window"](rlm.md) - treats context as an external state variable rather than a passive input tensor, enabling **O(log N)** retrieval through programmatic exploration.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=sk-your-key-here

# Interactive mode
python rlm_cli.py repl ../L1.md
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    USER QUERY                    │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│              ROOT LM (Depth=0)                   │
│  • Sees: metadata only (len, file info)          │
│  • Writes: Python code to explore context        │
│  • Uses: llm_query() for semantic analysis       │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│              REPL ENVIRONMENT                    │
│  • context: The full document (external state)   │
│  • RESULTS: Accumulated query results            │
│  • Executes code, returns stdout to Root LM      │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│            LEAF LM (Depth=1+)                    │
│  • Spawned by llm_query(instruction, chunk)      │
│  • Sees: Only the specific chunk                 │
│  • Returns: Semantic analysis to Root LM         │
└──────────────────────────────────────────────────┘
```

## CLI Usage

### Query Documents
```bash
python rlm_cli.py query L1.md "List all artists mentioned"
python rlm_cli.py query L1.md l2.md "What themes are common?"
```

### Extract Tracks (Music Documents)
```bash
python rlm_cli.py extract ../L1.md --output tracks.json
python rlm_cli.py extract ../L1.md ../l2.md --youtube
```

### Diff Documents
```bash
python rlm_cli.py diff ../L1.md ../l2.md
```

### Interactive REPL
```bash
python rlm_cli.py repl ../L1.md
# Then type queries interactively
```

## Python API

```python
from rlm_core import RLM

# Initialize
rlm = RLM()

# Load a large document
rlm.load_context("../L1.md")

# Query it (RLM writes code to explore)
result = rlm.query("Extract all track names and their functions")
print(result)
```

### Playlist Agent (Music-specific)

```python
from playlist_agent import PlaylistAgent

agent = PlaylistAgent()
agent.load_documents("../L1.md", "../l2.md")

# Extract structured track data
tracks = agent.extract_tracks()
for t in tracks:
    print(f"{t.artist} - {t.title} ({t.function})")

# Generate YouTube search links
playlist = agent.build_youtube_playlist()

# Export to JSON
agent.export_json("latent-radio-playlist.json")
```

## How It Works

1. **Context Loading**: Document loaded into Python variable (not LLM context)
2. **Root Query**: LLM receives only metadata (`context has 50,000 chars`)
3. **Code Synthesis**: LLM writes Python to explore (`print(context[:1000])`)
4. **Execution**: REPL runs code, captures output
5. **Semantic Sub-queries**: LLM calls `llm_query(instruction, chunk)` for meaning
6. **Aggregation**: Results stored in `RESULTS`, final in `FINAL`

This achieves **O(log N)** scaling for retrieval tasks - a 10M token document costs the same as a 10K token document for point queries.

## Files

- `rlm_core.py` - Core RLM engine with REPL and recursive primitives
- `playlist_agent.py` - Music document analyzer
- `rlm_cli.py` - Command-line interface
- `requirements.txt` - Python dependencies
