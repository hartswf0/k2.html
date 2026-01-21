#!/usr/bin/env python3
"""
RLM CLI - Command-line interface for the Recursive Language Model
==================================================================

Examples:
    # Process a single file
    python rlm_cli.py query L1.md "List all artists mentioned"
    
    # Extract tracks to JSON
    python rlm_cli.py extract L1.md l2.md --output tracks.json
    
    # Diff two documents
    python rlm_cli.py diff L1.md l2.md
    
    # Interactive REPL mode
    python rlm_cli.py repl L1.md
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rlm_core import RLM, RLMConfig
from playlist_agent import PlaylistAgent


def cmd_query(args):
    """Run a single query against documents"""
    rlm = RLM(verbose=not args.quiet)
    
    # Load context
    if len(args.files) == 1:
        rlm.load_context(args.files[0])
    else:
        # Multiple files
        combined = []
        for f in args.files:
            path = Path(f)
            content = path.read_text()
            combined.append(f"\n\n[FILE: {path.name}]\n{content}")
        rlm.env.context = "".join(combined)
        rlm.env.metadata = {"files": args.files, "total_chars": len(rlm.env.context)}
    
    result = rlm.query(args.query)
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result)


def cmd_extract(args):
    """Extract tracks from music documents"""
    agent = PlaylistAgent(verbose=not args.quiet)
    agent.load_documents(*args.files)
    
    tracks = agent.extract_tracks()
    
    if args.output:
        agent.export_json(args.output)
    else:
        print("\n📀 Extracted Tracks:")
        print("-" * 40)
        for t in tracks:
            print(f"  {t.id:02d}. {t.artist} - {t.title}")
            if t.function:
                print(f"      Function: {t.function}")
    
    if args.youtube:
        print("\n🔗 YouTube Links:")
        for p in agent.build_youtube_playlist():
            print(f"  {p['youtube_url']}")


def cmd_diff(args):
    """Diff two documents for track differences"""
    agent = PlaylistAgent(verbose=not args.quiet)
    
    result = agent.diff_documents(args.file1, args.file2)
    
    print("\n📊 Document Diff:")
    print("=" * 60)
    
    if "only_doc1" in result:
        print(f"\n🔵 Only in {Path(args.file1).name}:")
        for t in result.get("only_doc1", []):
            print(f"   - {t.get('artist', '?')} - {t.get('track', t.get('title', '?'))}")
    
    if "only_doc2" in result:
        print(f"\n🟢 Only in {Path(args.file2).name}:")
        for t in result.get("only_doc2", []):
            print(f"   - {t.get('artist', '?')} - {t.get('track', t.get('title', '?'))}")
    
    if "common" in result:
        print(f"\n⚪ In both documents:")
        for t in result.get("common", []):
            print(f"   - {t.get('artist', '?')} - {t.get('track', t.get('title', '?'))}")


def cmd_repl(args):
    """Interactive REPL mode"""
    rlm = RLM(verbose=True)
    
    # Load context
    if args.file:
        meta = rlm.load_context(args.file)
        print(f"📂 Loaded: {meta.get('size_chars', 0):,} chars from {args.file}")
    
    print("\n🔮 RLM Interactive Mode")
    print("   Type queries, or 'quit' to exit")
    print("   Use 'load <file>' to load a new context")
    print("-" * 40)
    
    while True:
        try:
            query = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if query.lower().startswith('load '):
            filepath = query[5:].strip()
            try:
                meta = rlm.load_context(filepath)
                print(f"📂 Loaded: {meta.get('size_chars', 0):,} chars")
            except Exception as e:
                print(f"❌ Error: {e}")
            continue
        
        if query.lower() == 'stats':
            print(json.dumps(rlm.get_stats(), indent=2))
            continue
        
        try:
            result = rlm.query(query)
            print("\n" + str(result))
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="RLM - Recursive Language Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Query command
    p_query = subparsers.add_parser("query", help="Run a query against documents")
    p_query.add_argument("files", nargs="+", help="Files to process")
    p_query.add_argument("query", help="The query to run")
    p_query.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    
    # Extract command
    p_extract = subparsers.add_parser("extract", help="Extract tracks from music documents")
    p_extract.add_argument("files", nargs="+", help="Files to process")
    p_extract.add_argument("-o", "--output", help="Output JSON file")
    p_extract.add_argument("-y", "--youtube", action="store_true", help="Show YouTube links")
    p_extract.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    
    # Diff command
    p_diff = subparsers.add_parser("diff", help="Diff two documents")
    p_diff.add_argument("file1", help="First document")
    p_diff.add_argument("file2", help="Second document")
    p_diff.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    
    # REPL command
    p_repl = subparsers.add_parser("repl", help="Interactive REPL mode")
    p_repl.add_argument("file", nargs="?", help="Initial file to load")
    
    args = parser.parse_args()
    
    if args.command == "query":
        cmd_query(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "repl":
        cmd_repl(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
