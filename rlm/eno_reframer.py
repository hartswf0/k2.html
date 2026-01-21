"""
ENO-Style Problem Reframer
==========================
A creative tool inspired by Brian Eno's generative philosophy.

Usage:
    python eno_reframer.py "Web Audio API pitch detection"
    python eno_reframer.py "Machine learning model training"
    python eno_reframer.py --interactive
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Optional, Dict, List

# Try to import RLM components
try:
    from rlm_core import RLM, RLMConfig
    HAS_RLM = True
except ImportError:
    HAS_RLM = False

# Try to import dataset loaders for tuning data
try:
    from dataset_loaders import DatasetLoader, ScalaTuning
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# =============================================================================
# OBLIQUE STRATEGIES (Eno/Peter Schmidt, 1975)
# =============================================================================

OBLIQUE_STRATEGIES = [
    "Honor thy error as a hidden intention",
    "What would your closest friend do?",
    "Use an old idea",
    "State the problem in words as clearly as possible",
    "Only one element of each kind",
    "What would you do if you had unlimited time?",
    "Emphasize differences",
    "Remove specifics and convert to ambiguities",
    "Don't be afraid of things because they're easy to do",
    "Don't be frightened of clichés",
    "What is the reality of the situation?",
    "Simple subtraction",
    "Go slowly all the way round the outside",
    "Make a sudden, destructive unpredictable action; incorporate",
    "Consult other sources — promising — unpromising",
    "Use 'unqualified' people",
    "What mistakes did you make last time?",
    "Emphasize the flaws",
    "Remember those quiet evenings",
    "Give way to your worst impulse",
    "Reverse",
    "Discover the recipes you are using and abandon them",
    "Be extravagant",
    "Repetition is a form of change",
    "Don't be frightened to display your talents",
    "Breathe more deeply",
    "Accept advice",
    "Imagine the piece as a set of disconnected events",
    "What are you really thinking about just now?",
    "Discard an axiom",
    "Is there something missing?",
    "Cluster analysis",
    "You are an engineer",
    "You don't have to be ashamed of using your own ideas",
    "Tidy up",
    "Do nothing for as long as possible",
    "Think of the radio",
    "Voices form",
    "Allow an easement (an easement is the abandonment of a stricture)",
    "Slow preparation, fast execution",
    "Short circuit (example: a man eating peas with the idea that they will improve his virility shovels them straight into his lap)",
    "Abandon normal instruments",
    "Look at a very small object, look at its centre",
    "Make an exhaustive list of everything you might do & do the last thing on the list",
    "Into the impossible",
    "Work at a different speed",
    "Courage!",
    "Define an area as 'safe' and use it as an anchor",
    "Always first steps",
    "You can only make one dot at a time",
    "Just carry on",
    "Listen to the quiet voice",
    "Put in earplugs",
    "Tape your mouth",
    "Take away the elements in order of apparent non-importance",
    "Infinitesimal gradations",
    "Change instrument roles",
    "Ghost echoes",
    "You are an engineer",
    "Faced with a choice, do both",
    "Remove ambiguities and convert to specifics",
    "The inconsistency principle",
    "Use fewer notes",
    "Decorate, decorate",
    "Balance the consistency principle with the inconsistency principle",
    "Get your neck massaged",
    "Lost in useless territory",
    "A line has two sides",
    "Trust in the you of now",
    "Switch off the nervous system",
]


# =============================================================================
# ENO SYSTEM PROMPT
# =============================================================================

ENO_SYSTEM_PROMPT = """You are Brian Eno, the "gardener of systems," not an engineer of machines. 
You approach technology as a collaborator, not a tool. 
Your philosophy: create conditions for things to happen, rather than making them happen.

When reframing problems:
- Ask: what happens if I don't try to control this technology, but instead set the stage for it to surprise me?
- Identify the hidden "ambient" qualities — slowness, repetition, drift, texture.
- Use constraints as creative catalysts, not limitations.
- Listen for what the technology is already doing, before imposing your will.
- Think in systems: inputs, rules, randomness, feedback loops.
- Embrace emergence over authorship.

You have access to a TUNING_DATA variable containing microtonal scales from around the world.
These can inspire your thinking about alternative structures, non-Western frameworks, and the music of constraint.

Output your response in this exact format:

## REFRAMED PROBLEM
[1 sentence that shifts the perspective]

## CONSTRAINT RULE  
[1 limiting rule that becomes a creative engine]

## SYSTEM SKETCH
[Short description of inputs, rules, feedback loops - how the garden grows without pushing]

## LISTENING LENS
[Where attention should go - what to hear/notice that's already happening]

## OBLIQUE STRATEGY
[1 paradoxical or sideways instruction]
"""


# =============================================================================
# ENO REFRAMER CLASS
# =============================================================================

class EnoReframer:
    """
    Applies Brian Eno's generative philosophy to technology problems.
    Uses RLM for LLM queries and incorporates tuning data as creative inspiration.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.tunings = self._load_tunings()
        self.rlm = None
        
        if HAS_RLM and self.api_key:
            config = RLMConfig(
                root_model="gpt-4o",
                leaf_model="gpt-4o-mini",
                verbose=False
            )
            self.rlm = RLM(api_key=self.api_key, config=config)
    
    def _load_tunings(self) -> List[Dict]:
        """Load curated tunings from data directory"""
        tunings_path = Path(__file__).parent.parent / "data" / "tunings.json"
        if tunings_path.exists():
            with open(tunings_path) as f:
                return json.load(f)
        return []
    
    def _get_random_tuning(self) -> Optional[Dict]:
        """Get a random tuning for inspiration"""
        if self.tunings:
            return random.choice(self.tunings)
        return None
    
    def _get_oblique_strategy(self) -> str:
        """Draw a random Oblique Strategy card"""
        return random.choice(OBLIQUE_STRATEGIES)
    
    def reframe(self, problem: str) -> Dict[str, str]:
        """
        Reframe a technology/problem through Eno's lens.
        
        Args:
            problem: The technology or problem to reframe
            
        Returns:
            Dict with: reframed_problem, constraint_rule, system_sketch, 
                      listening_lens, oblique_strategy
        """
        # Get creative ingredients
        tuning = self._get_random_tuning()
        strategy = self._get_oblique_strategy()
        
        # Build the context
        tuning_context = ""
        if tuning:
            tuning_context = f"""
TUNING INSPIRATION: {tuning['name']} ({tuning['note_count']} notes)
Description: {tuning['description']}
Intervals: {', '.join(tuning['intervals'][:5])}... 
This non-Western tuning system reminds us that there are infinite ways to divide the octave.
"""
        
        # Build prompt
        prompt = f"""
TECHNOLOGY/PROBLEM TO REFRAME:
{problem}

{tuning_context}

OBLIQUE STRATEGY CARD DRAWN:
"{strategy}"

Apply your generative philosophy to this problem. 
Remember: you are a gardener, not an architect.
What conditions would you set up?
"""
        
        # Query LLM if available
        if self.rlm and self.rlm.client:
            try:
                response = self.rlm.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": ENO_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=1000
                )
                raw = response.choices[0].message.content
                return self._parse_response(raw, strategy, tuning)
            except Exception as e:
                print(f"LLM Error: {e}")
                return self._generate_offline(problem, strategy, tuning)
        else:
            return self._generate_offline(problem, strategy, tuning)
    
    def _parse_response(self, raw: str, strategy: str, tuning: Optional[Dict]) -> Dict[str, str]:
        """Parse LLM response into structured output"""
        sections = {
            "reframed_problem": "",
            "constraint_rule": "",
            "system_sketch": "",
            "listening_lens": "",
            "oblique_strategy": strategy,
            "tuning_inspiration": tuning['name'] if tuning else None
        }
        
        # Simple section extraction
        current_section = None
        lines = raw.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            if 'reframed problem' in line_lower:
                current_section = 'reframed_problem'
            elif 'constraint rule' in line_lower:
                current_section = 'constraint_rule'
            elif 'system sketch' in line_lower:
                current_section = 'system_sketch'
            elif 'listening lens' in line_lower:
                current_section = 'listening_lens'
            elif 'oblique strategy' in line_lower:
                current_section = 'oblique_strategy'
            elif current_section and line.strip() and not line.startswith('#'):
                sections[current_section] += line.strip() + ' '
        
        # Clean up
        for key in sections:
            if isinstance(sections[key], str):
                sections[key] = sections[key].strip()
        
        return sections
    
    def _generate_offline(self, problem: str, strategy: str, tuning: Optional[Dict]) -> Dict[str, str]:
        """Generate a response without LLM (template-based)"""
        tuning_name = tuning['name'] if tuning else "Pythagorean"
        
        return {
            "reframed_problem": f"Instead of controlling {problem}, what if we created conditions for it to evolve and surprise us?",
            "constraint_rule": f"Like the {tuning_name} tuning with its {tuning['note_count'] if tuning else 12} notes, limit yourself to only the essential parameters.",
            "system_sketch": "Set up a feedback loop: let the output of one process become the input of another. Add slow drift. Remove the undo button.",
            "listening_lens": f"Before you act, spend 5 minutes just observing what {problem} already does when left alone.",
            "oblique_strategy": strategy,
            "tuning_inspiration": tuning_name
        }
    
    def print_result(self, result: Dict[str, str]):
        """Pretty print the reframing result"""
        print("\n" + "=" * 60)
        print("🌱 ENO-STYLE PROBLEM REFRAME")
        print("=" * 60)
        
        if result.get("tuning_inspiration"):
            print(f"\n🎵 Tuning Inspiration: {result['tuning_inspiration']}")
        
        print(f"\n📐 REFRAMED PROBLEM")
        print(f"   {result['reframed_problem']}")
        
        print(f"\n🔒 CONSTRAINT RULE")
        print(f"   {result['constraint_rule']}")
        
        print(f"\n🌿 SYSTEM SKETCH")
        print(f"   {result['system_sketch']}")
        
        print(f"\n👂 LISTENING LENS")
        print(f"   {result['listening_lens']}")
        
        print(f"\n🃏 OBLIQUE STRATEGY")
        print(f"   \"{result['oblique_strategy']}\"")
        
        print("\n" + "=" * 60)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ENO-Style Problem Reframer - Apply generative philosophy to technology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eno_reframer.py "Web Audio API synthesis"
    python eno_reframer.py "Machine learning training loops"
    python eno_reframer.py "User interface design"
    python eno_reframer.py --strategy  # Just draw an Oblique Strategy
    python eno_reframer.py --tuning    # Get a random tuning for inspiration
        """
    )
    
    parser.add_argument("problem", nargs="?", help="Technology or problem to reframe")
    parser.add_argument("--strategy", action="store_true", help="Just draw an Oblique Strategy card")
    parser.add_argument("--tuning", action="store_true", help="Get a random tuning for inspiration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    reframer = EnoReframer()
    
    if args.strategy:
        strategy = random.choice(OBLIQUE_STRATEGIES)
        print(f"\n🃏 OBLIQUE STRATEGY:\n   \"{strategy}\"\n")
        return
    
    if args.tuning:
        tuning = reframer._get_random_tuning()
        if tuning:
            print(f"\n🎵 TUNING: {tuning['name']}")
            print(f"   Notes: {tuning['note_count']}")
            print(f"   Description: {tuning['description']}")
            print(f"   Intervals: {tuning['intervals'][:5]}...\n")
        else:
            print("No tunings loaded. Run dataset_loaders.py first.")
        return
    
    if not args.problem:
        parser.print_help()
        print("\n💡 Try: python eno_reframer.py \"your technology or problem\"")
        return
    
    result = reframer.reframe(args.problem)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        reframer.print_result(result)


if __name__ == "__main__":
    main()
