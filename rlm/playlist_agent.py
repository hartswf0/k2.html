"""
Playlist Agent - RLM-powered music document analyzer
=====================================================
Uses the RLM architecture to extract tracks, artists, and build playlists
from large music documentation files.

Specialized for the Latent Radio / Forager Crate format.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from rlm_core import RLM, RLMConfig


@dataclass
class Track:
    """A track extracted from a document"""
    id: int
    artist: str
    title: str
    album: Optional[str] = None
    function: Optional[str] = None
    operator: Optional[str] = None
    bpm: Optional[int] = None
    key: Optional[str] = None
    youtube_search: Optional[str] = None
    source_file: Optional[str] = None
    

class PlaylistAgent:
    """
    An RLM-based agent specialized for music document analysis.
    
    Capabilities:
    - Extract tracks from prose descriptions
    - Identify artists, albums, functions
    - Generate YouTube search queries
    - Diff between document versions
    - Build structured playlists
    """
    
    PLAYLIST_PROMPT = """You are a music document analyzer. Your task is to extract structured track information.

For each track mentioned, identify:
- Artist name
- Track title  
- Album (if mentioned)
- Function/Role in the document (e.g., "Seismic Anchor", "Atmospheric Floor")
- BPM (if mentioned)
- Key signature (if mentioned)

Output as JSON list:
```json
[
  {"artist": "...", "title": "...", "album": "...", "function": "...", "bpm": null, "key": null},
  ...
]
```
"""
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        config = RLMConfig(
            root_model="gpt-4o",
            leaf_model="gpt-4o-mini",
            max_depth=2,
            chunk_size=6000,
            verbose=verbose
        )
        self.rlm = RLM(api_key=api_key, config=config)
        self.tracks: List[Track] = []
        
    def load_documents(self, *paths: str) -> Dict:
        """Load one or more documents into the RLM context"""
        if len(paths) == 1:
            path = Path(paths[0])
            if path.is_dir():
                return self.rlm.load_context(str(path), "*.md")
            else:
                return self.rlm.load_context(str(path))
        else:
            # Multiple files - concatenate
            combined = []
            for p in paths:
                path = Path(p)
                content = path.read_text()
                combined.append(f"\n\n{'='*60}\n[FILE: {path.name}]\n{'='*60}\n\n{content}")
            
            self.rlm.env.context = "".join(combined)
            self.rlm.env.metadata = {
                "files": [str(p) for p in paths],
                "total_chars": len(self.rlm.env.context)
            }
            return self.rlm.env.metadata
    
    def extract_tracks(self) -> List[Track]:
        """Extract all tracks from the loaded documents"""
        query = """
        Find ALL music tracks/songs mentioned in this document.
        For each track, extract:
        - Artist name
        - Track/song title
        - Album name (if present)
        - Function or operator role (if described, like "Seismic Anchor" or "Atmospheric Floor")
        - BPM (if mentioned)
        - Musical key (if mentioned)
        
        Return as a JSON array. Be exhaustive - find EVERY track mentioned.
        Store the result in FINAL as a JSON string.
        """
        
        result = self.rlm.query(query)
        
        try:
            # Parse JSON result
            if isinstance(result, str):
                # Try to extract JSON from the result
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(result)
            else:
                data = result
            
            # Convert to Track objects
            self.tracks = []
            for i, item in enumerate(data, 1):
                track = Track(
                    id=i,
                    artist=item.get('artist', 'Unknown'),
                    title=item.get('title', item.get('track', 'Unknown')),
                    album=item.get('album'),
                    function=item.get('function'),
                    operator=item.get('operator'),
                    bpm=item.get('bpm'),
                    key=item.get('key'),
                    youtube_search=f"{item.get('artist', '')} {item.get('title', item.get('track', ''))}"
                )
                self.tracks.append(track)
                
            return self.tracks
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw result: {result[:500]}")
            return []
    
    def diff_documents(self, doc1_path: str, doc2_path: str) -> Dict:
        """
        Compare two documents and find:
        - Tracks unique to doc1
        - Tracks unique to doc2
        - Tracks in both (common)
        """
        # Load both documents
        doc1 = Path(doc1_path).read_text()
        doc2 = Path(doc2_path).read_text()
        
        combined = f"""
DOCUMENT 1: {Path(doc1_path).name}
{'='*60}
{doc1}

{'='*60}
DOCUMENT 2: {Path(doc2_path).name}
{'='*60}
{doc2}
"""
        self.rlm.env.context = combined
        self.rlm.env.metadata = {
            "doc1": doc1_path,
            "doc2": doc2_path,
            "total_chars": len(combined)
        }
        
        query = """
        Compare DOCUMENT 1 and DOCUMENT 2.
        
        Find all tracks/artists mentioned in each.
        Identify:
        1. Tracks ONLY in Document 1 (not in Document 2)
        2. Tracks ONLY in Document 2 (not in Document 1)
        3. Tracks in BOTH documents
        
        Return as JSON:
        {
            "only_doc1": [{"artist": "...", "track": "..."}],
            "only_doc2": [{"artist": "...", "track": "..."}],
            "common": [{"artist": "...", "track": "..."}]
        }
        
        Store in FINAL as a JSON string.
        """
        
        result = self.rlm.query(query)
        
        try:
            if isinstance(result, str):
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return json.loads(result)
            return result
        except:
            return {"raw": result}
    
    def build_youtube_playlist(self) -> List[Dict]:
        """Generate YouTube search links for all extracted tracks"""
        if not self.tracks:
            self.extract_tracks()
        
        playlist = []
        for track in self.tracks:
            query = f"{track.artist} {track.title}"
            playlist.append({
                "id": track.id,
                "artist": track.artist,
                "title": track.title,
                "function": track.function,
                "youtube_url": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            })
        
        return playlist
    
    def export_json(self, filepath: str):
        """Export tracks to JSON file"""
        data = {
            "extracted_at": self.rlm.env.metadata.get("loaded_at"),
            "source": self.rlm.env.metadata,
            "tracks": [
                {
                    "id": t.id,
                    "artist": t.artist,
                    "title": t.title,
                    "album": t.album,
                    "function": t.function,
                    "operator": t.operator,
                    "bpm": t.bpm,
                    "key": t.key,
                    "youtube_search": t.youtube_search
                }
                for t in self.tracks
            ]
        }
        
        Path(filepath).write_text(json.dumps(data, indent=2))
        print(f"Exported {len(self.tracks)} tracks to {filepath}")
    
    def interactive_query(self, question: str) -> str:
        """Ask any question about the loaded documents"""
        return self.rlm.query(question)


def main():
    """Demo: Process the K2 music documents"""
    print("🎵 Playlist Agent - RLM-powered music analyzer")
    print("=" * 60)
    
    # Check for documents
    k2_path = Path("/Users/gaia/K2")
    md_files = list(k2_path.glob("*.md"))
    
    print(f"\nFound {len(md_files)} markdown files in K2:")
    for f in md_files:
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    
    print("\nUsage:")
    print("  agent = PlaylistAgent()")
    print("  agent.load_documents('L1.md', 'l2.md')")
    print("  tracks = agent.extract_tracks()")
    print("  playlist = agent.build_youtube_playlist()")
    print("  agent.export_json('playlist.json')")


if __name__ == "__main__":
    main()
