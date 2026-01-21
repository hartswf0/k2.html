"""
RLM Local - Recursive Language Model for IDE/Antigravity
=========================================================
This version works WITHOUT external APIs.
The IDE's built-in LLM (Antigravity) acts as both Root and Leaf LM.

Usage - Interactive with IDE:
    1. Run: python rlm_local.py load L1.md l2.md
    2. Copy the output context info
    3. Ask the IDE LLM to analyze with RLM prompts
    4. Run code snippets the LLM generates
    5. Feed results back to continue
    
Or use the helper functions to explore manually.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RLMContext:
    """The loaded context and metadata"""
    content: str
    metadata: Dict
    results: Dict
    

class RLMLocal:
    """
    Local RLM that works with IDE's built-in LLM.
    
    The IDE LLM (you, Antigravity) acts as:
    - Root LM: Planning and code synthesis
    - Leaf LM: Semantic analysis of chunks
    
    This class provides the REPL environment and helper tools.
    """
    
    def __init__(self):
        self.context = ""
        self.metadata = {}
        self.RESULTS = {}
        self._files = []
        
    def load(self, *filepaths: str) -> str:
        """Load files into context. Returns summary for LLM."""
        combined = []
        file_info = []
        
        for fp in filepaths:
            path = Path(fp)
            if not path.exists():
                # Try relative to K2
                path = Path("/Users/gaia/K2") / fp
            
            content = path.read_text()
            marker = f"\n{'='*60}\n[FILE: {path.name}]\n{'='*60}\n"
            combined.append(marker + content)
            file_info.append({
                "name": path.name,
                "chars": len(content),
                "lines": content.count('\n') + 1
            })
            self._files.append(str(path))
        
        self.context = "".join(combined)
        self.metadata = {
            "files": file_info,
            "total_chars": len(self.context),
            "total_lines": self.context.count('\n') + 1
        }
        
        return self._get_context_summary()
    
    def _get_context_summary(self) -> str:
        """Get a summary suitable for LLM prompt"""
        files_str = "\n".join(
            f"  - {f['name']}: {f['chars']:,} chars, {f['lines']} lines"
            for f in self.metadata.get('files', [])
        )
        return f"""
📂 RLM Context Loaded
{'='*40}
Total: {self.metadata['total_chars']:,} characters
Files:
{files_str}

The content is stored in variable `context`.
Use peek(), search(), slice_context() to explore.
"""
    
    # === REPL Tools (call these from IDE) ===
    
    def peek(self, start: int = 0, length: int = 1000) -> str:
        """Peek at a section of the context"""
        chunk = self.context[start:start + length]
        return f"[chars {start}-{start+length}]:\n{chunk}"
    
    def peek_end(self, length: int = 1000) -> str:
        """Peek at the end of the context"""
        chunk = self.context[-length:]
        return f"[last {length} chars]:\n{chunk}"
    
    def search(self, pattern: str, max_results: int = 10) -> List[Dict]:
        """Search for regex pattern, return matches with positions"""
        matches = []
        for m in re.finditer(pattern, self.context, re.IGNORECASE):
            if len(matches) >= max_results:
                break
            matches.append({
                "position": m.start(),
                "match": m.group()[:100],
                "context": self.context[max(0,m.start()-50):m.end()+50]
            })
        return matches
    
    def find_all(self, pattern: str) -> List[str]:
        """Find all matches of a pattern"""
        return re.findall(pattern, self.context, re.IGNORECASE)
    
    def slice_context(self, start: int, end: int) -> str:
        """Get a slice of context"""
        return self.context[start:end]
    
    def get_window(self, position: int, window: int = 2000) -> str:
        """Get context window around a position"""
        start = max(0, position - window // 2)
        end = min(len(self.context), position + window // 2)
        return self.context[start:end]
    
    def count(self, pattern: str) -> int:
        """Count occurrences of a pattern"""
        return len(re.findall(pattern, self.context, re.IGNORECASE))
    
    def extract_between(self, start_pattern: str, end_pattern: str) -> List[str]:
        """Extract text between two patterns"""
        pattern = f"{start_pattern}(.*?){end_pattern}"
        return re.findall(pattern, self.context, re.DOTALL | re.IGNORECASE)
    
    def find_sections(self, header_pattern: str = r'^#+\s+(.+)$') -> List[Dict]:
        """Find markdown sections"""
        sections = []
        for m in re.finditer(header_pattern, self.context, re.MULTILINE):
            sections.append({
                "title": m.group(1),
                "position": m.start(),
                "level": m.group().count('#')
            })
        return sections
    
    def store_result(self, key: str, value):
        """Store a result for later use"""
        self.RESULTS[key] = value
        return f"Stored in RESULTS['{key}']"
    
    def get_results(self) -> Dict:
        """Get all stored results"""
        return self.RESULTS
    
    # === Track Extraction Helpers ===
    
    def extract_tracks_pattern(self) -> List[Dict]:
        """
        Extract tracks using common patterns found in music docs.
        Pattern: Artist – "Track" or Artist - Track
        """
        tracks = []
        
        # Pattern 1: Artist – "Track" (fancy quotes)
        for m in re.finditer(r'([A-Z][^–—\n]+?)\s*[–—]\s*["""]([^"""]+)["""]', self.context):
            tracks.append({"artist": m.group(1).strip(), "track": m.group(2).strip()})
        
        # Pattern 2: Subject: Artist – "Track"
        for m in re.finditer(r'Subject:\s*([^–\n]+)\s*–\s*["""]([^"""]+)["""]', self.context):
            tracks.append({"artist": m.group(1).strip(), "track": m.group(2).strip()})
        
        # Pattern 3: Artist - "Track" in quotes
        for m in re.finditer(r'([A-Z][a-zA-Z\s]+)\s*[-–]\s*"([^"]+)"', self.context):
            tracks.append({"artist": m.group(1).strip(), "track": m.group(2).strip()})
        
        # Dedupe
        seen = set()
        unique = []
        for t in tracks:
            key = (t['artist'].lower(), t['track'].lower())
            if key not in seen:
                seen.add(key)
                unique.append(t)
        
        return unique
    
    def build_playlist_json(self, tracks: List[Dict]) -> str:
        """Build a playlist JSON from extracted tracks"""
        playlist = []
        for i, t in enumerate(tracks, 1):
            playlist.append({
                "id": i,
                "artist": t.get('artist'),
                "track": t.get('track'),
                "youtube_search": f"{t.get('artist', '')} {t.get('track', '')}".strip()
            })
        return json.dumps(playlist, indent=2)


# === Quick-start functions ===

def load(*files):
    """Quick load - returns RLM instance"""
    rlm = RLMLocal()
    print(rlm.load(*files))
    return rlm


def demo():
    """Demo with K2 music docs"""
    rlm = load("L1.md", "l2.md")
    
    print("\n🔍 Peeking at start:")
    print(rlm.peek(0, 500))
    
    print("\n🔍 Searching for 'Senyawa':")
    print(rlm.search("Senyawa"))
    
    print("\n🎵 Extracting tracks:")
    tracks = rlm.extract_tracks_pattern()
    for t in tracks[:10]:
        print(f"  {t['artist']} - {t['track']}")
    
    return rlm


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo()
        elif sys.argv[1] == "load":
            rlm = load(*sys.argv[2:])
            print("\nRLM ready. Use rlm.peek(), rlm.search(), etc.")
        else:
            # Treat args as files
            rlm = load(*sys.argv[1:])
    else:
        print("""
RLM Local - IDE-native Recursive Language Model
================================================

Usage:
  python rlm_local.py load L1.md l2.md   # Load files
  python rlm_local.py demo               # Run demo

Or in Python:
  from rlm_local import load
  rlm = load("L1.md", "l2.md")
  print(rlm.peek())
  print(rlm.search("artist name"))
  tracks = rlm.extract_tracks_pattern()
""")
