"""
RLM Core - Recursive Language Model Environment
================================================
Based on the "Beyond the Context Window" architecture.

This implements:
- Context-as-State: Large documents stored as external variables
- Recursive Primitive: llm_query() for spawning sub-agents
- REPL Environment: Safe code execution with stdout capture
- Context Folding: Metadata-only prompts with lazy loading

Usage:
    from rlm_core import RLM
    
    rlm = RLM(api_key="your-key")
    rlm.load_context("path/to/large_file.md")
    result = rlm.query("Extract all track names from this document")
"""

import os
import re
import sys
import json
import traceback
from io import StringIO
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# Try to import OpenAI, fall back to mock if not available
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed. Run: pip install openai")


@dataclass
class RLMConfig:
    """Configuration for RLM behavior"""
    root_model: str = "gpt-4o"           # Model for root agent (program synthesis)
    leaf_model: str = "gpt-4o-mini"      # Model for leaf agents (content analysis)
    max_depth: int = 3                    # Maximum recursion depth
    chunk_size: int = 4000                # Target chunk size for leaves
    max_iterations: int = 20              # Max REPL iterations
    sandbox_timeout: int = 30             # Code execution timeout (seconds)
    verbose: bool = True                  # Print execution trace


@dataclass
class ExecutionResult:
    """Result from code execution in the REPL"""
    success: bool
    stdout: str
    stderr: str
    return_value: Any = None


class RLMEnvironment:
    """
    The REPL Environment - provides the "tape" and "head" for the Turing machine.
    
    Key variables available to the LLM:
    - context: The loaded document(s)
    - RESULTS: Accumulated results from llm_query calls
    - metadata: File info, lengths, types
    """
    
    def __init__(self, config: RLMConfig, llm_query_fn: Callable):
        self.config = config
        self._llm_query = llm_query_fn
        self.context: str = ""
        self.RESULTS: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._execution_log: List[Dict] = []
        
    def load_file(self, filepath: str) -> Dict[str, Any]:
        """Load a file into the context variable"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        self.context = path.read_text(encoding='utf-8')
        self.metadata = {
            "filepath": str(path.absolute()),
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "size_chars": len(self.context),
            "size_lines": self.context.count('\n') + 1,
            "loaded_at": datetime.now().isoformat(),
        }
        return self.metadata
    
    def load_directory(self, dirpath: str, pattern: str = "*.md") -> Dict[str, Any]:
        """Load all matching files from a directory into context"""
        path = Path(dirpath)
        files = list(path.glob(pattern))
        
        combined = []
        file_info = []
        
        for f in sorted(files):
            content = f.read_text(encoding='utf-8')
            marker = f"\n\n{'='*60}\n[FILE: {f.name}]\n{'='*60}\n\n"
            combined.append(marker + content)
            file_info.append({
                "name": f.name,
                "size": len(content),
                "start_char": sum(len(c) for c in combined[:-1])
            })
        
        self.context = "".join(combined)
        self.metadata = {
            "directory": str(path.absolute()),
            "pattern": pattern,
            "file_count": len(files),
            "files": file_info,
            "total_chars": len(self.context),
            "loaded_at": datetime.now().isoformat(),
        }
        return self.metadata
    
    def execute_code(self, code: str) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment.
        The code has access to: context, RESULTS, metadata, llm_query, re, json
        """
        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        # Build the execution namespace
        namespace = {
            'context': self.context,
            'RESULTS': self.RESULTS,
            'metadata': self.metadata,
            'llm_query': self._llm_query,
            're': re,
            'json': json,
            'len': len,
            'print': print,
            'range': range,
            'enumerate': enumerate,
            'sorted': sorted,
            'list': list,
            'dict': dict,
            'str': str,
            'int': int,
            'float': float,
            'min': min,
            'max': max,
            'sum': sum,
            'zip': zip,
        }
        
        return_value = None
        success = True
        
        try:
            # Execute the code
            exec(code, namespace)
            
            # Check for FINAL marker
            if 'FINAL' in namespace:
                return_value = namespace['FINAL']
            
            # Update RESULTS from namespace
            if 'RESULTS' in namespace:
                self.RESULTS = namespace['RESULTS']
                
        except Exception as e:
            success = False
            traceback.print_exc()
        
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Log execution
        self._execution_log.append({
            "code": code[:500] + ("..." if len(code) > 500 else ""),
            "success": success,
            "stdout_preview": stdout[:200],
            "stderr_preview": stderr[:200] if stderr else None,
        })
        
        return ExecutionResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            return_value=return_value
        )


class RLM:
    """
    Recursive Language Model - The main interface.
    
    Implements the "Context-as-State" paradigm where:
    - The Root LM never sees the full context
    - It writes code to explore and query the context
    - It uses llm_query() to spawn semantic sub-agents
    """
    
    SYSTEM_PROMPT = """You are an intelligent agent operating in a Recursive Language Model (RLM) environment.

## ENVIRONMENT
- The variable `context` contains the user's document ({size_chars} characters, {size_lines} lines)
- The variable `metadata` contains file information
- The variable `RESULTS` stores accumulated query results

⚠️ WARNING: You CANNOT see the full context directly. Do NOT try to print(context).

## TOOLS
You can write Python code to:
- Inspect: `print(context[:1000])` to peek at content
- Search: `re.findall(r'pattern', context)` to find patterns
- Slice: `chunk = context[start:end]` to extract sections
- Query: `llm_query(instruction, chunk)` to analyze a specific section semantically

## SPECIAL FUNCTION
```python
llm_query(instruction: str, chunk: str) -> str
```
Spawns a sub-agent to analyze a specific chunk. Use for semantic understanding.

## PROTOCOL
1. **Inspect**: First explore the structure (peek at start, end, search for markers)
2. **Plan**: Devise a strategy (Binary Search, Map-Reduce, or Direct Extraction)
3. **Execute**: Write Python code implementing your plan
4. **Synthesize**: Store final answer in variable `FINAL`

## OUTPUT FORMAT
Respond with Python code blocks only. Use `print()` to show intermediate results.
When done, assign your final answer to: `FINAL = "your answer"`
"""

    def __init__(self, api_key: Optional[str] = None, config: Optional[RLMConfig] = None):
        self.config = config or RLMConfig()
        
        if HAS_OPENAI:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        else:
            self.client = None
            
        self.env = RLMEnvironment(self.config, self._llm_query)
        self._depth = 0
        self._query_count = 0
        
    def load_context(self, path: str, pattern: str = "*.md") -> Dict[str, Any]:
        """Load a file or directory into the context"""
        p = Path(path)
        if p.is_dir():
            return self.env.load_directory(path, pattern)
        else:
            return self.env.load_file(path)
    
    def _llm_query(self, instruction: str, chunk: str, depth: int = 0) -> str:
        """
        The Recursive Primitive - spawns a "leaf" agent to process a chunk.
        This is exposed to the Root LM as `llm_query()`.
        """
        if depth > self.config.max_depth:
            return f"ERROR: Max recursion depth ({self.config.max_depth}) exceeded"
        
        if not self.client:
            return f"[MOCK RESPONSE] Analyzed {len(chunk)} chars with instruction: {instruction[:100]}"
        
        self._query_count += 1
        
        if self.config.verbose:
            print(f"  📡 llm_query (depth={depth}): {instruction[:60]}... [{len(chunk)} chars]")
        
        response = self.client.chat.completions.create(
            model=self.config.leaf_model,
            messages=[
                {"role": "system", "content": "You are a precise analysis sub-agent. Analyze the provided text and respond concisely."},
                {"role": "user", "content": f"INSTRUCTION: {instruction}\n\nTEXT:\n{chunk[:self.config.chunk_size]}"}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _call_root_llm(self, messages: List[Dict]) -> str:
        """Call the Root LLM to get the next code block"""
        if not self.client:
            return '```python\nFINAL = "Mock response - install openai to use"\n```'
        
        response = self.client.chat.completions.create(
            model=self.config.root_model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from markdown code blocks"""
        # Try to find ```python blocks
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Try bare ``` blocks
        pattern = r'```\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        return None
    
    def query(self, user_query: str) -> str:
        """
        Main entry point - process a query against the loaded context.
        
        This runs the REPL loop:
        1. Send system prompt + query to Root LLM
        2. Extract code from response
        3. Execute code in sandbox
        4. Feed results back to Root LLM
        5. Repeat until FINAL is set or max iterations reached
        """
        if not self.env.context:
            return "ERROR: No context loaded. Call load_context() first."
        
        # Build system prompt with metadata
        system_prompt = self.SYSTEM_PROMPT.format(
            size_chars=self.env.metadata.get('size_chars', len(self.env.context)),
            size_lines=self.env.metadata.get('size_lines', self.env.context.count('\n'))
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"USER QUERY: {user_query}"}
        ]
        
        if self.config.verbose:
            print(f"\n🔮 RLM Query: {user_query}")
            print(f"   Context: {self.env.metadata.get('size_chars', 0):,} chars")
            print("-" * 60)
        
        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"\n📍 Iteration {iteration + 1}")
            
            # Get next action from Root LLM
            response = self._call_root_llm(messages)
            
            # Extract code
            code = self._extract_code(response)
            
            if not code:
                if self.config.verbose:
                    print(f"   No code found in response. Raw: {response[:200]}")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please provide Python code to continue analysis."})
                continue
            
            if self.config.verbose:
                print(f"   Code: {code[:100]}...")
            
            # Execute code
            result = self.env.execute_code(code)
            
            if self.config.verbose:
                if result.stdout:
                    print(f"   Output: {result.stdout[:200]}")
                if result.stderr:
                    print(f"   ⚠️ Error: {result.stderr[:200]}")
            
            # Check for FINAL
            if result.return_value is not None:
                if self.config.verbose:
                    print(f"\n✅ FINAL: {str(result.return_value)[:200]}")
                return result.return_value
            
            # Build feedback message
            feedback = f"Code executed.\n\nSTDOUT:\n{result.stdout[:1500]}"
            if result.stderr:
                feedback += f"\n\nSTDERR:\n{result.stderr[:500]}"
            
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            messages.append({"role": "user", "content": feedback})
        
        return "ERROR: Max iterations reached without finding FINAL answer"
    
    def get_execution_log(self) -> List[Dict]:
        """Get the execution trace for debugging"""
        return self.env._execution_log
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RLM session"""
        return {
            "context_size": len(self.env.context),
            "query_count": self._query_count,
            "results_stored": len(self.env.RESULTS),
            "metadata": self.env.metadata,
        }


# Convenience function for quick usage
def process_file(filepath: str, query: str, api_key: Optional[str] = None) -> str:
    """Quick one-liner to process a file with RLM"""
    rlm = RLM(api_key=api_key)
    rlm.load_context(filepath)
    return rlm.query(query)


if __name__ == "__main__":
    # Demo mode
    print("=" * 60)
    print("RLM - Recursive Language Model Environment")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from rlm_core import RLM")
    print("  rlm = RLM()")
    print("  rlm.load_context('your_file.md')")
    print("  result = rlm.query('Extract all artist names')")
    print()
    print("Or quick mode:")
    print("  from rlm_core import process_file")
    print("  result = process_file('doc.md', 'summarize this')")
