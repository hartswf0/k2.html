"""
MIDI Generator - Pure Python MIDI File Writer
==============================================
No external dependencies. Generates Standard MIDI Files (SMF).

Usage:
    from midi_writer import MIDIWriter
    
    midi = MIDIWriter(bpm=120)
    midi.note(pitch=60, start=0, duration=0.5, velocity=100)
    midi.note(pitch=64, start=0.5, duration=0.5, velocity=90)
    midi.save("output.mid")
"""

import struct
from pathlib import Path
from typing import List, Tuple, Optional


class MIDIWriter:
    """
    Pure Python MIDI file writer.
    Creates Type 0 Standard MIDI Files (single track).
    """
    
    def __init__(self, bpm: int = 120, ticks_per_beat: int = 480):
        self.bpm = bpm
        self.ticks_per_beat = ticks_per_beat
        self.events: List[Tuple[int, bytes]] = []  # (tick, event_bytes)
        self._current_tick = 0
    
    def _variable_length(self, value: int) -> bytes:
        """Encode integer as MIDI variable-length quantity"""
        result = []
        result.append(value & 0x7F)
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))
    
    def _seconds_to_ticks(self, seconds: float) -> int:
        """Convert seconds to MIDI ticks"""
        return int(seconds * self.ticks_per_beat * self.bpm / 60)
    
    def _beats_to_ticks(self, beats: float) -> int:
        """Convert beats to MIDI ticks"""
        return int(beats * self.ticks_per_beat)
    
    def note(self, pitch: int, start: float, duration: float, 
             velocity: int = 100, channel: int = 0, use_beats: bool = False):
        """
        Add a note event.
        
        Args:
            pitch: MIDI note number (60 = Middle C)
            start: Start time in seconds (or beats if use_beats=True)
            duration: Duration in seconds (or beats if use_beats=True)
            velocity: Note velocity (0-127)
            channel: MIDI channel (0-15)
            use_beats: If True, start/duration are in beats, not seconds
        """
        if use_beats:
            start_tick = self._beats_to_ticks(start)
            dur_ticks = self._beats_to_ticks(duration)
        else:
            start_tick = self._seconds_to_ticks(start)
            dur_ticks = self._seconds_to_ticks(duration)
        
        end_tick = start_tick + dur_ticks
        
        # Note On event (0x90 = Note On, channel 0)
        note_on = bytes([0x90 | channel, pitch, velocity])
        self.events.append((start_tick, note_on))
        
        # Note Off event (0x80 = Note Off)
        note_off = bytes([0x80 | channel, pitch, 0])
        self.events.append((end_tick, note_off))
    
    def set_tempo(self, bpm: int):
        """Add tempo meta event"""
        self.bpm = bpm
        # Tempo in microseconds per beat
        tempo = int(60_000_000 / bpm)
        tempo_bytes = tempo.to_bytes(3, 'big')
        # Meta event: FF 51 03 <tempo>
        event = bytes([0xFF, 0x51, 0x03]) + tempo_bytes
        self.events.append((0, event))
    
    def set_instrument(self, program: int, channel: int = 0):
        """Set instrument (Program Change)"""
        event = bytes([0xC0 | channel, program])
        self.events.append((0, event))
    
    def _build_track(self) -> bytes:
        """Build the track chunk"""
        # Sort events by tick time
        sorted_events = sorted(self.events, key=lambda x: x[0])
        
        track_data = bytearray()
        prev_tick = 0
        
        for tick, event_bytes in sorted_events:
            delta = tick - prev_tick
            track_data.extend(self._variable_length(delta))
            track_data.extend(event_bytes)
            prev_tick = tick
        
        # End of track meta event
        track_data.extend(self._variable_length(0))
        track_data.extend(bytes([0xFF, 0x2F, 0x00]))
        
        return bytes(track_data)
    
    def save(self, filepath: str):
        """Save MIDI file to disk"""
        path = Path(filepath)
        
        # Build header chunk
        # "MThd" + length(6) + format(0) + tracks(1) + division
        header = b'MThd'
        header += struct.pack('>I', 6)  # Header length
        header += struct.pack('>H', 0)  # Format 0 (single track)
        header += struct.pack('>H', 1)  # Number of tracks
        header += struct.pack('>H', self.ticks_per_beat)  # Ticks per beat
        
        # Build track chunk
        track_data = self._build_track()
        track = b'MTrk'
        track += struct.pack('>I', len(track_data))
        track += track_data
        
        # Write file
        with open(path, 'wb') as f:
            f.write(header)
            f.write(track)
        
        return str(path)
    
    def to_bytes(self) -> bytes:
        """Get MIDI file as bytes (for streaming)"""
        header = b'MThd'
        header += struct.pack('>I', 6)
        header += struct.pack('>H', 0)
        header += struct.pack('>H', 1)
        header += struct.pack('>H', self.ticks_per_beat)
        
        track_data = self._build_track()
        track = b'MTrk'
        track += struct.pack('>I', len(track_data))
        track += track_data
        
        return header + track


def create_scale_midi(intervals_cents: List[float], 
                      root_note: int = 60,
                      bpm: int = 120,
                      note_duration: float = 0.5) -> MIDIWriter:
    """
    Create MIDI file from Scala intervals.
    
    Since MIDI is 12-TET, we map microtonal cents to nearest semitone.
    For true microtonal, you'd use pitch bend or MPE.
    
    Args:
        intervals_cents: List of intervals in cents from tunings.json
        root_note: MIDI note for root (60 = Middle C)
        bpm: Tempo
        note_duration: Duration per note in beats
    """
    midi = MIDIWriter(bpm=bpm)
    midi.set_tempo(bpm)
    
    # Play root
    midi.note(root_note, start=0, duration=note_duration, use_beats=True)
    
    # Play each interval
    for i, cents in enumerate(intervals_cents):
        # Convert cents to semitones (100 cents = 1 semitone)
        if isinstance(cents, str):
            # Handle ratios like "3/2"
            if '/' in cents:
                num, denom = cents.split('/')
                ratio = float(num) / float(denom)
                import math
                cents = 1200 * math.log2(ratio)
            else:
                cents = float(cents)
        
        semitones = round(cents / 100)
        pitch = root_note + semitones
        
        # Clamp to valid MIDI range
        pitch = max(0, min(127, pitch))
        
        start_beat = (i + 1) * note_duration
        midi.note(pitch, start=start_beat, duration=note_duration, use_beats=True)
    
    return midi


def create_sequence_midi(sequence: List[Optional[int]],
                         intervals_cents: List[float],
                         root_note: int = 60,
                         bpm: int = 120,
                         step_duration: float = 0.25) -> MIDIWriter:
    """
    Create MIDI from a generated sequence (from composer.py).
    
    Args:
        sequence: List of note indices (None = rest)
        intervals_cents: The tuning's intervals
        root_note: MIDI root note
        bpm: Tempo
        step_duration: Duration per step in beats (0.25 = 16th note)
    """
    midi = MIDIWriter(bpm=bpm)
    midi.set_tempo(bpm)
    
    for step, note_idx in enumerate(sequence):
        if note_idx is None:
            continue
        
        # Get cents for this scale degree
        if note_idx < len(intervals_cents):
            cents = intervals_cents[note_idx]
        else:
            # Octave up
            octave = note_idx // len(intervals_cents)
            degree = note_idx % len(intervals_cents)
            cents = intervals_cents[degree] if degree < len(intervals_cents) else 0
            cents += octave * 1200
        
        # Convert to semitones
        if isinstance(cents, str):
            if '/' in cents:
                num, denom = cents.split('/')
                ratio = float(num) / float(denom)
                import math
                cents = 1200 * math.log2(ratio)
            else:
                cents = float(cents)
        
        semitones = round(cents / 100)
        pitch = root_note + semitones
        pitch = max(0, min(127, pitch))
        
        start = step * step_duration
        midi.note(pitch, start=start, duration=step_duration * 0.9, use_beats=True)
    
    return midi


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Test: create a simple scale
    print("Creating test MIDI files...")
    
    # C Major scale
    midi = MIDIWriter(bpm=120)
    midi.set_tempo(120)
    midi.set_instrument(0)  # Piano
    
    c_major = [0, 2, 4, 5, 7, 9, 11, 12]  # Semitones
    for i, semi in enumerate(c_major):
        midi.note(60 + semi, start=i * 0.5, duration=0.45, use_beats=True)
    
    output = "../data/test_scale.mid"
    midi.save(output)
    print(f"✓ Saved: {output}")
    
    # Test with Slendro tuning
    slendro_cents = [231.17, 498.04, 701.96, 968.83, 1200]
    slendro_midi = create_scale_midi(slendro_cents, root_note=60, bpm=100)
    slendro_output = "../data/slendro_scale.mid"
    slendro_midi.save(slendro_output)
    print(f"✓ Saved: {slendro_output}")
    
    # Test sequence
    test_seq = [0, None, 2, None, 1, 3, None, 4, 0, None, 2, 1, None, 3, 4, 0]
    seq_midi = create_sequence_midi(test_seq, slendro_cents, bpm=110)
    seq_output = "../data/slendro_sequence.mid"
    seq_midi.save(seq_output)
    print(f"✓ Saved: {seq_output}")
    
    print("\nDone! Open these files in any DAW or MIDI player.")
