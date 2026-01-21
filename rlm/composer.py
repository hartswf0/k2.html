"""
LATENT COMPOSER - RLM-Powered Generative Music
==============================================
Uses the Latent Engine theory (from L1.md/l2.md) to generate compositions
that follow the "Thinking Operators" framework.

The system reads:
- L1.md/l2.md: Musicological theory (Operators, Axes, Transitions)
- sample-crate-120.json: Sampleable tracks
- tunings.json: Microtonal scales from Scala archive

And generates:
- Setlists with operator-based logic
- Sequences with BPM/key transitions
- Microtonal compositions using Scala tunings

Usage:
    python composer.py generate --theme "seismic dread"
    python composer.py setlist --operators "anchor,slingshot,cavern"
    python composer.py sequence --tuning slendro --steps 16
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Try to import RLM
try:
    from rlm_core import RLM, RLMConfig
    HAS_RLM = True
except ImportError:
    HAS_RLM = False

# Import MIDI writer
try:
    from midi_writer import MIDIWriter, create_sequence_midi, create_scale_midi
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False


# =============================================================================
# THINKING OPERATORS (from L1.md theory)
# =============================================================================

@dataclass
class Operator:
    """A Thinking Operator from the Latent Engine"""
    name: str
    function: str
    axis: str  # Mass, Velocity, Space, Time, Geometry, Biology
    sonic_signature: str
    bpm_range: tuple
    transition_logic: str


OPERATORS = {
    "anchor": Operator(
        name="The Seismic Anchor",
        function="Lithic Resonance / Structural Stress",
        axis="Mass",
        sonic_signature="Low-frequency resonance, physical distortion, acoustic friction",
        bpm_range=(80, 150),
        transition_logic="High friction → builds tension for release"
    ),
    "slingshot": Operator(
        name="The Keplerian Speed-Run",
        function="Variable Velocity / Orbital Acceleration",
        axis="Velocity", 
        sonic_signature="Accelerating rhythms, polyrhythms, algorithmic sequencing",
        bpm_range=(130, 180),
        transition_logic="Zero friction → launches from previous mass"
    ),
    "floor": Operator(
        name="The Atmospheric Floor",
        function="Material Dust / High-Vertical Filtering",
        axis="Space",
        sonic_signature="Room tone, high-frequency air, hollow middle",
        bpm_range=(80, 105),
        transition_logic="Deposits listener in detailed room"
    ),
    "cavern": Operator(
        name="The Deep Cavern",
        function="Seismic Dread / The Zimmer Scale",
        axis="Time",
        sonic_signature="Sub-bass below 40Hz, viscous time-delays",
        bpm_range=(70, 90),
        transition_logic="Vertical expansion → suppression"
    ),
    "pulse": Operator(
        name="The Geometric Pulse",
        function="Base-8 Geometry / Celestial Timing",
        axis="Geometry",
        sonic_signature="Odd meters, Snaketime, mantra repetition",
        bpm_range=(100, 130),
        transition_logic="Resets internal clock → removes 4/4 expectation"
    ),
    "lung": Operator(
        name="The Biological Lung",
        function="Respiratory Rate / Bio-Logic",
        axis="Biology",
        sonic_signature="Breathing patterns, organic oscillation",
        bpm_range=(60, 90),
        transition_logic="Entrains listener's physiology"
    )
}


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_sample_crate() -> List[Dict]:
    """Load sample-crate-120.json"""
    crate_path = Path(__file__).parent.parent / "sample-crate-120.json"
    if crate_path.exists():
        with open(crate_path) as f:
            data = json.load(f)
            return data.get('tracks', [])
    return []


def load_tunings() -> List[Dict]:
    """Load tunings.json from data directory"""
    tunings_path = Path(__file__).parent.parent / "data" / "tunings.json"
    if tunings_path.exists():
        with open(tunings_path) as f:
            return json.load(f)
    return []


def load_latent_radio() -> List[Dict]:
    """Load latent-radio-120.json"""
    radio_path = Path(__file__).parent.parent / "latent-radio-120.json"
    if radio_path.exists():
        with open(radio_path) as f:
            data = json.load(f)
            return data.get('tracks', data) if isinstance(data, dict) else data
    return []


# =============================================================================
# COMPOSITION GENERATOR
# =============================================================================

class LatentComposer:
    """
    Generative music composition using RLM + Latent Engine theory.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.samples = load_sample_crate()
        self.tunings = load_tunings()
        self.tracks = load_latent_radio()
        self.rlm = None
        
        if HAS_RLM and self.api_key:
            config = RLMConfig(verbose=False)
            self.rlm = RLM(api_key=self.api_key, config=config)
    
    def generate_setlist(self, 
                         operators: List[str] = None,
                         duration_minutes: int = 60) -> List[Dict]:
        """
        Generate a setlist based on Thinking Operators.
        
        Args:
            operators: List of operator keys (e.g., ["anchor", "slingshot", "cavern"])
            duration_minutes: Target duration
            
        Returns:
            List of track dicts with transitions
        """
        if not operators:
            # Default journey: Mass → Velocity → Space → Time → Geometry → Biology
            operators = ["anchor", "slingshot", "floor", "cavern", "pulse", "lung"]
        
        setlist = []
        samples = self.samples.copy()
        
        for i, op_key in enumerate(operators):
            op = OPERATORS.get(op_key)
            if not op:
                continue
            
            # Find matching samples by BPM range or category
            candidates = [s for s in samples 
                          if s.get('category') in self._get_categories_for_axis(op.axis)]
            
            if not candidates:
                candidates = samples
            
            # Pick one
            if candidates:
                track = random.choice(candidates)
                samples.remove(track)  # Don't repeat
                
                setlist.append({
                    "position": i + 1,
                    "operator": op.name,
                    "axis": op.axis,
                    "track": f"{track['artist']} - {track['track']}",
                    "year": track.get('year'),
                    "sample_note": track.get('sample_note'),
                    "transition_logic": op.transition_logic,
                    "bpm_target": sum(op.bpm_range) // 2
                })
        
        return setlist
    
    def _get_categories_for_axis(self, axis: str) -> List[str]:
        """Map axes to sample categories"""
        mapping = {
            "Mass": ["funk", "rock", "world"],
            "Velocity": ["electronic", "hiphop"],
            "Space": ["jazz", "soundtrack"],
            "Time": ["soundtrack", "electronic"],
            "Geometry": ["jazz", "world"],
            "Biology": ["jazz", "funk"]
        }
        return mapping.get(axis, [])
    
    def generate_sequence(self, 
                          tuning_name: str = "slendro",
                          steps: int = 16,
                          operator: str = "pulse") -> Dict:
        """
        Generate a microtonal sequence using a Scala tuning.
        
        Args:
            tuning_name: Name of tuning from tunings.json
            steps: Number of sequence steps
            operator: Operator to guide the pattern
            
        Returns:
            Dict with sequence data for scala-live.html
        """
        # Find tuning
        tuning = None
        for t in self.tunings:
            if t['name'].lower() == tuning_name.lower():
                tuning = t
                break
        
        if not tuning:
            tuning = {"name": "12-TET", "intervals": list(range(1, 13)), "note_count": 12}
        
        op = OPERATORS.get(operator, OPERATORS["pulse"])
        note_count = tuning.get('note_count', len(tuning.get('intervals', [])))
        
        # Generate pattern based on operator logic
        sequence = []
        
        if op.axis == "Geometry":
            # Geometric patterns: base-8, odd meters
            for i in range(steps):
                if i % 3 == 0 or i % 5 == 0:  # Snaketime-ish
                    sequence.append(random.randint(0, note_count - 1))
                else:
                    sequence.append(None)
        
        elif op.axis == "Biology":
            # Breathing pattern: inhale (ascending), exhale (descending)
            cycle = steps // 4
            for i in range(steps):
                phase = (i // cycle) % 4
                if phase == 0:  # Inhale
                    sequence.append(min(i % note_count, note_count - 1))
                elif phase == 2:  # Exhale
                    sequence.append(max(note_count - 1 - (i % note_count), 0))
                else:  # Pause
                    sequence.append(None)
        
        elif op.axis == "Mass":
            # Heavy, low notes with friction
            for i in range(steps):
                if random.random() > 0.4:
                    sequence.append(random.randint(0, note_count // 3))  # Low range
                else:
                    sequence.append(None)
        
        elif op.axis == "Velocity":
            # Accelerating density
            for i in range(steps):
                density = (i / steps) * 0.8 + 0.2
                if random.random() < density:
                    sequence.append(random.randint(0, note_count - 1))
                else:
                    sequence.append(None)
        
        else:
            # Default: random with 40% density
            for i in range(steps):
                if random.random() > 0.6:
                    sequence.append(random.randint(0, note_count - 1))
                else:
                    sequence.append(None)
        
        return {
            "tuning": tuning['name'],
            "note_count": note_count,
            "operator": op.name,
            "axis": op.axis,
            "bpm": sum(op.bpm_range) // 2,
            "steps": steps,
            "sequence": sequence,
            "pattern_logic": op.function
        }
    
    def compose_with_rlm(self, prompt: str) -> str:
        """
        Use RLM to compose based on the full musicological context.
        Loads L1.md and uses it to reason about composition.
        """
        if not self.rlm:
            return self._compose_offline(prompt)
        
        # Load musicological context
        l1_path = Path(__file__).parent.parent / "L1.md"
        if l1_path.exists():
            self.rlm.load_context(str(l1_path))
        
        composition_query = f"""
Based on the Latent Engine theory in this document, compose a setlist for:
{prompt}

Extract the relevant "Thinking Operators" and their transition logic.
For each position in the setlist, provide:
1. The Operator name and function
2. A track suggestion that fits (can be from the document or your knowledge)
3. The BPM and key target
4. The transition logic to the next track

Output as structured JSON.
"""
        
        return self.rlm.query(composition_query)
    
    def _compose_offline(self, prompt: str) -> str:
        """Offline composition without LLM"""
        # Generate based on keywords in prompt
        ops = []
        prompt_lower = prompt.lower()
        
        if "heavy" in prompt_lower or "mass" in prompt_lower:
            ops.append("anchor")
        if "fast" in prompt_lower or "velocity" in prompt_lower:
            ops.append("slingshot")
        if "space" in prompt_lower or "air" in prompt_lower:
            ops.append("floor")
        if "dread" in prompt_lower or "dark" in prompt_lower:
            ops.append("cavern")
        if "rhythm" in prompt_lower or "geometry" in prompt_lower:
            ops.append("pulse")
        if "breath" in prompt_lower or "organic" in prompt_lower:
            ops.append("lung")
        
        if not ops:
            ops = ["anchor", "slingshot", "cavern", "lung"]
        
        setlist = self.generate_setlist(operators=ops)
        return json.dumps(setlist, indent=2)
    
    def export_for_scala_live(self, sequence: Dict) -> str:
        """Export sequence to JavaScript for scala-live.html"""
        js_code = f"""
// Generated by Latent Composer
// Operator: {sequence['operator']} ({sequence['axis']})
// Tuning: {sequence['tuning']} ({sequence['note_count']} notes)
// BPM: {sequence['bpm']}

const GENERATED_SEQUENCE = {json.dumps(sequence['sequence'])};
const GENERATED_BPM = {sequence['bpm']};

// To use: paste into browser console on scala-live.html
sequence = GENERATED_SEQUENCE;
bpm = GENERATED_BPM;
document.getElementById('bpm-slider').value = bpm;
document.getElementById('bpm-val').textContent = bpm;
renderSequencer();
"""
        return js_code


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LATENT COMPOSER - RLM-Powered Generative Music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python composer.py setlist --operators anchor,slingshot,cavern
    python composer.py sequence --tuning slendro --operator pulse
    python composer.py compose "heavy dread into geometric release"
    python composer.py operators  # List all operators
        """
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Setlist command
    setlist_p = subparsers.add_parser("setlist", help="Generate operator-based setlist")
    setlist_p.add_argument("--operators", help="Comma-separated operator keys")
    setlist_p.add_argument("--duration", type=int, default=60, help="Target minutes")
    
    # Sequence command
    seq_p = subparsers.add_parser("sequence", help="Generate microtonal sequence")
    seq_p.add_argument("--tuning", default="slendro", help="Tuning name")
    seq_p.add_argument("--operator", default="pulse", help="Operator to guide pattern")
    seq_p.add_argument("--steps", type=int, default=16, help="Number of steps")
    seq_p.add_argument("--export", action="store_true", help="Export JS for scala-live.html")
    seq_p.add_argument("--midi", help="Export MIDI file to specified path")
    
    # Compose command
    compose_p = subparsers.add_parser("compose", help="Free-form composition with RLM")
    compose_p.add_argument("prompt", help="Composition prompt")
    
    # List operators
    ops_p = subparsers.add_parser("operators", help="List all Thinking Operators")
    
    args = parser.parse_args()
    
    composer = LatentComposer()
    
    if args.command == "operators":
        print("\n" + "=" * 60)
        print("THINKING OPERATORS (from Latent Engine)")
        print("=" * 60)
        for key, op in OPERATORS.items():
            print(f"\n  [{key}] {op.name}")
            print(f"      Axis: {op.axis}")
            print(f"      Function: {op.function}")
            print(f"      BPM Range: {op.bpm_range[0]}-{op.bpm_range[1]}")
            print(f"      Sonic: {op.sonic_signature[:60]}...")
    
    elif args.command == "setlist":
        ops = args.operators.split(",") if args.operators else None
        setlist = composer.generate_setlist(operators=ops, duration_minutes=args.duration)
        
        print("\n" + "=" * 60)
        print("GENERATED SETLIST")
        print("=" * 60)
        for track in setlist:
            print(f"\n  {track['position']}. [{track['axis']}] {track['operator']}")
            print(f"     → {track['track']} ({track['year']})")
            print(f"     BPM: ~{track['bpm_target']} | {track['sample_note']}")
            print(f"     Transition: {track['transition_logic']}")
    
    elif args.command == "sequence":
        seq = composer.generate_sequence(
            tuning_name=args.tuning,
            operator=args.operator,
            steps=args.steps
        )
        
        if args.midi:
            if HAS_MIDI:
                # Get tuning intervals
                tuning = None
                for t in composer.tunings:
                    if t['name'].lower() == args.tuning.lower():
                        tuning = t
                        break
                intervals = tuning['intervals'] if tuning else [100 * i for i in range(1, 13)]
                
                midi = create_sequence_midi(
                    sequence=seq['sequence'],
                    intervals_cents=intervals,
                    root_note=60,
                    bpm=seq['bpm'],
                    step_duration=0.25
                )
                midi.save(args.midi)
                print(f"✓ Saved MIDI: {args.midi}")
                print(f"  Tuning: {seq['tuning']} | BPM: {seq['bpm']}")
                print(f"  Operator: {seq['operator']}")
            else:
                print("Error: midi_writer.py not found")
        elif args.export:
            print(composer.export_for_scala_live(seq))
        else:
            print("\n" + "=" * 60)
            print(f"GENERATED SEQUENCE: {seq['tuning']} × {seq['operator']}")
            print("=" * 60)
            print(f"  Tuning: {seq['tuning']} ({seq['note_count']} notes)")
            print(f"  Operator: {seq['operator']} [{seq['axis']}]")
            print(f"  BPM: {seq['bpm']}")
            print(f"  Pattern Logic: {seq['pattern_logic']}")
            print(f"\n  Sequence: {seq['sequence']}")
            print(f"\n  Visual:")
            for i, note in enumerate(seq['sequence']):
                marker = f"[{note:2d}]" if note is not None else "[ · ]"
                print(f"    Step {i+1:2d}: {marker}")
    
    elif args.command == "compose":
        result = composer.compose_with_rlm(args.prompt)
        print("\n" + "=" * 60)
        print("COMPOSED WITH RLM")
        print("=" * 60)
        print(result)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
