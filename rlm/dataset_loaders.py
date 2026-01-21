"""
Dataset Loaders for RLM
=======================
Integrates external audio/music datasets into the RLM environment.

Supported Datasets:
- NSynth: Parametric music note dataset from Magenta
- WebAudioFont: Browser-playable instrument samples (JSON-encoded)
- Scala: Microtonal tuning archive (.scl to JSON)
- FSD50K: Environmental sound classification

Usage:
    from dataset_loaders import DatasetLoader
    
    loader = DatasetLoader(cache_dir="./data")
    
    # Load NSynth metadata
    nsynth_df = loader.load_nsynth(subset="test")
    
    # Load a WebAudioFont instrument
    piano = loader.load_webaudiofont("0000_JCLive_sf2_file")
    
    # Load a Scala tuning
    tuning = loader.load_scala_tuning("zeus22")
    
    # Load FSD50K ground truth
    fsd_df = loader.load_fsd50k()
"""

import os
import json
import tarfile
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  Pandas not installed. DataFrames will not be available.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class ScalaTuning:
    """Represents a parsed Scala tuning file"""
    name: str
    description: str
    note_count: int
    intervals: List[str]  # Cents or ratios
    
    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "note_count": self.note_count,
            "intervals": self.intervals
        }
    
    def to_cents(self) -> List[float]:
        """Convert all intervals to cents values"""
        result = [0.0]  # Root is always 0 cents
        for interval in self.intervals:
            if '/' in interval:
                # Ratio format (e.g., "3/2")
                num, denom = interval.split('/')
                ratio = float(num) / float(denom)
                cents = 1200 * (ratio ** (1/12) - 1) * 12  # Approximate
                # Actually: cents = 1200 * log2(ratio)
                import math
                cents = 1200 * math.log2(ratio)
            else:
                # Already in cents
                cents = float(interval.replace('.', '', 1).isdigit() and interval or interval)
                try:
                    cents = float(interval)
                except:
                    cents = 0.0
            result.append(cents)
        return result


@dataclass 
class WebAudioFontZone:
    """A single zone from a WebAudioFont instrument"""
    key_range_low: int
    key_range_high: int
    original_pitch: int
    sample_rate: int
    has_audio: bool
    ahdsr: Optional[Dict] = None


@dataclass
class WebAudioFontInstrument:
    """Parsed WebAudioFont instrument data"""
    name: str
    zones: List[WebAudioFontZone]
    
    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "zone_count": len(self.zones),
            "zones": [
                {
                    "keyRange": f"{z.key_range_low}-{z.key_range_high}",
                    "originalPitch": z.original_pitch,
                    "sampleRate": z.sample_rate,
                    "hasAudio": z.has_audio
                }
                for z in self.zones
            ]
        }


class DatasetLoader:
    """
    Unified loader for external audio datasets.
    Downloads and caches data locally for RLM processing.
    """
    
    # Dataset URLs
    NSYNTH_TEST_URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
    NSYNTH_TRAIN_URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
    WEBAUDIOFONT_BASE = "https://surikov.github.io/webaudiofontdata/sound/"
    SCALA_ARCHIVE_URL = "http://www.huygens-fokker.org/docs/scales.zip"
    FSD50K_GROUND_TRUTH_URL = "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip"
    
    def __init__(self, cache_dir: str = "../data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _download_file(self, url: str, filename: str) -> Path:
        """Download a file if not already cached"""
        filepath = self.cache_dir / filename
        if not filepath.exists():
            print(f"📥 Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"✅ Downloaded to {filepath}")
        return filepath
    
    # =========================================================================
    # NSYNTH - Parametric Note Dataset
    # =========================================================================
    
    def load_nsynth(self, subset: str = "test", download_audio: bool = False) -> Any:
        """
        Load NSynth dataset metadata.
        
        Args:
            subset: "test" (300MB) or "train" (20GB+)
            download_audio: If True, also downloads WAV files
            
        Returns:
            DataFrame with note metadata (if pandas) or dict
        """
        url = self.NSYNTH_TEST_URL if subset == "test" else self.NSYNTH_TRAIN_URL
        filename = f"nsynth-{subset}.jsonwav.tar.gz"
        extract_dir = self.cache_dir / f"nsynth-{subset}"
        
        # Download archive
        archive_path = self._download_file(url, filename)
        
        # Extract JSON only (skip audio unless requested)
        if not extract_dir.exists():
            print(f"📦 Extracting {filename}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                if download_audio:
                    tar.extractall(self.cache_dir)
                else:
                    # Extract only examples.json
                    for member in tar.getmembers():
                        if member.name.endswith("examples.json"):
                            tar.extract(member, self.cache_dir)
                            break
        
        # Load JSON
        json_path = extract_dir / "examples.json"
        if not json_path.exists():
            # Try alternate path structure
            json_path = self.cache_dir / f"nsynth-{subset}" / "examples.json"
        
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
            
            if HAS_PANDAS:
                return pd.DataFrame.from_dict(data, orient='index')
            return data
        else:
            raise FileNotFoundError(f"Could not find examples.json in {extract_dir}")
    
    def filter_nsynth(self, df: Any, 
                      instrument_family: Optional[str] = None,
                      qualities: Optional[List[str]] = None) -> Any:
        """
        Filter NSynth data by instrument family and/or qualities.
        
        Qualities index: 0=bright, 1=dark, 2=distortion, 3=fast_decay,
                        4=long_release, 5=multiphonic, 6=nonlinear_env,
                        7=percussive, 8=reverb, 9=tempo-synced
        """
        if not HAS_PANDAS:
            raise RuntimeError("Pandas required for filtering. pip install pandas")
        
        result = df.copy()
        
        if instrument_family:
            result = result[result['instrument_family_str'] == instrument_family]
        
        if qualities:
            quality_map = {
                'bright': 0, 'dark': 1, 'distortion': 2, 'fast_decay': 3,
                'long_release': 4, 'multiphonic': 5, 'nonlinear_env': 6,
                'percussive': 7, 'reverb': 8, 'tempo-synced': 9
            }
            for q in qualities:
                idx = quality_map.get(q.lower())
                if idx is not None:
                    result = result[result['qualities'].apply(lambda x: x[idx] == 1)]
        
        return result
    
    # =========================================================================
    # WEBAUDIOFONT - Browser Instruments
    # =========================================================================
    
    def _js_to_json(self, js_str: str) -> str:
        """Convert JavaScript object literal to valid JSON"""
        import re
        # Quote unquoted keys: {keyName: -> {"keyName":
        result = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', js_str)
        # Handle single quotes -> double quotes (carefully)
        result = result.replace("'", '"')
        return result
    
    def load_webaudiofont(self, instrument_name: str) -> WebAudioFontInstrument:
        """
        Load a WebAudioFont instrument definition.
        
        Args:
            instrument_name: e.g., "0250_SoundBlasterOld_sf2" (Electric Piano)
            
        Returns:
            WebAudioFontInstrument with zone data
        """
        url = f"{self.WEBAUDIOFONT_BASE}{instrument_name}.js"
        
        if HAS_REQUESTS:
            response = requests.get(url)
            js_content = response.text
        else:
            with urllib.request.urlopen(url) as response:
                js_content = response.read().decode('utf-8')
        
        # Parse JS to JSON (strip variable declaration)
        start_index = js_content.find('{')
        end_index = js_content.rfind('}') + 1
        
        if start_index == -1 or end_index == 0:
            raise ValueError(f"Could not parse instrument data from {url}")
        
        json_str = js_content[start_index:end_index]
        
        # Convert JS object literal to JSON
        json_str = self._js_to_json(json_str)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Fallback: extract basic info via regex
            print(f"⚠️ JSON parse failed, using fallback extraction: {e}")
            data = {"name": instrument_name, "zones": []}
        
        # Convert to structured format
        zones = []
        for zone in data.get('zones', []):
            zones.append(WebAudioFontZone(
                key_range_low=zone.get('keyRangeLow', 0),
                key_range_high=zone.get('keyRangeHigh', 127),
                original_pitch=zone.get('originalPitch', 60),
                sample_rate=zone.get('sampleRate', 44100),
                has_audio=bool(zone.get('file')),
                ahdsr=zone.get('ahdsr')
            ))
        
        return WebAudioFontInstrument(
            name=data.get('name', instrument_name),
            zones=zones
        )
    
    def list_webaudiofont_instruments(self) -> List[str]:
        """Return a list of common WebAudioFont instrument IDs"""
        # These are known good instruments from the library
        return [
            "0000_JCLive_sf2_file",      # Acoustic Grand Piano
            "0240_JCLive_sf2_file",      # Harpsichord
            "0250_JCLive_sf2_file",      # Electric Piano
            "0320_JCLive_sf2_file",      # Acoustic Bass
            "0400_JCLive_sf2_file",      # Violin
            "0460_JCLive_sf2_file",      # Harp
            "0560_JCLive_sf2_file",      # Trumpet
            "0680_JCLive_sf2_file",      # Oboe
            "0730_JCLive_sf2_file",      # Flute
            "1040_JCLive_sf2_file",      # Steel Drums
        ]
    
    # =========================================================================
    # SCALA - Microtonal Tunings
    # =========================================================================
    
    def load_scala_archive(self) -> Path:
        """Download and extract the Scala tuning archive"""
        archive_path = self._download_file(self.SCALA_ARCHIVE_URL, "scales.zip")
        return archive_path
    
    def load_scala_tuning(self, scale_name: str) -> ScalaTuning:
        """
        Load a specific Scala tuning file.
        
        Args:
            scale_name: e.g., "zeus22" or "zeus22.scl"
            
        Returns:
            ScalaTuning object with parsed intervals
        """
        if not scale_name.endswith('.scl'):
            scale_name += '.scl'
        
        archive_path = self.load_scala_archive()
        
        with zipfile.ZipFile(archive_path, 'r') as z:
            # Search for the file (might be in subdirectory)
            matching = [n for n in z.namelist() if n.endswith(scale_name)]
            
            if not matching:
                raise FileNotFoundError(f"Scale '{scale_name}' not found in archive")
            
            with z.open(matching[0]) as f:
                lines = f.read().decode('utf-8', errors='ignore').splitlines()
        
        # Parse SCL format
        description = ""
        note_count = 0
        intervals = []
        
        data_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('!'):
                if line.startswith('!') and not description:
                    description = line[1:].strip()
                continue
            data_lines.append(line)
        
        if len(data_lines) >= 2:
            description = data_lines[0] if not description else description
            try:
                note_count = int(data_lines[1])
            except:
                note_count = 0
            
            for interval_line in data_lines[2:]:
                # Remove inline comments
                value = interval_line.split('!')[0].strip()
                if value:
                    intervals.append(value)
        
        return ScalaTuning(
            name=scale_name.replace('.scl', ''),
            description=description,
            note_count=note_count,
            intervals=intervals
        )
    
    def list_scala_tunings(self, limit: int = 50) -> List[str]:
        """List available Scala tunings from the archive"""
        archive_path = self.load_scala_archive()
        
        with zipfile.ZipFile(archive_path, 'r') as z:
            scl_files = [n for n in z.namelist() if n.endswith('.scl')]
            return sorted(scl_files)[:limit]
    
    # =========================================================================
    # FSD50K - Environmental Sounds
    # =========================================================================
    
    def load_fsd50k(self) -> Any:
        """
        Load FSD50K ground truth metadata.
        
        Returns:
            DataFrame with sound labels (if pandas) or dict
        """
        filename = "FSD50K.ground_truth.zip"
        archive_path = self._download_file(self.FSD50K_GROUND_TRUTH_URL, filename)
        
        extract_dir = self.cache_dir / "FSD50K.ground_truth"
        
        if not extract_dir.exists():
            print(f"📦 Extracting {filename}...")
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(self.cache_dir)
        
        # Load training CSV
        train_csv = extract_dir / "dev.csv"
        if not train_csv.exists():
            train_csv = extract_dir / "train.csv"
        
        if HAS_PANDAS and train_csv.exists():
            return pd.read_csv(train_csv)
        elif train_csv.exists():
            # Manual CSV parsing
            with open(train_csv, 'r') as f:
                lines = f.readlines()
                headers = lines[0].strip().split(',')
                data = []
                for line in lines[1:]:
                    data.append(dict(zip(headers, line.strip().split(','))))
                return data
        else:
            raise FileNotFoundError(f"Could not find ground truth CSV in {extract_dir}")
    
    def filter_fsd50k(self, df: Any, labels: List[str]) -> Any:
        """Filter FSD50K by label keywords (OR logic)"""
        if not HAS_PANDAS:
            raise RuntimeError("Pandas required for filtering")
        
        mask = df['labels'].str.contains('|'.join(labels), case=False, na=False)
        return df[mask]


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RLM Dataset Loader")
    parser.add_argument("dataset", choices=["nsynth", "webaudiofont", "scala", "fsd50k"],
                       help="Dataset to load")
    parser.add_argument("--cache", default="./data", help="Cache directory")
    parser.add_argument("--item", help="Specific item (instrument name, scale name, etc.)")
    parser.add_argument("--list", action="store_true", help="List available items")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    loader = DatasetLoader(cache_dir=args.cache)
    
    if args.dataset == "nsynth":
        if args.list:
            print("NSynth subsets: test, train")
        else:
            df = loader.load_nsynth(subset="test")
            print(f"Loaded {len(df)} notes from NSynth")
            if args.output:
                df.to_json(args.output, orient='records', indent=2)
    
    elif args.dataset == "webaudiofont":
        if args.list:
            for i in loader.list_webaudiofont_instruments():
                print(i)
        elif args.item:
            inst = loader.load_webaudiofont(args.item)
            print(json.dumps(inst.to_json(), indent=2))
        else:
            print("Specify --item or --list")
    
    elif args.dataset == "scala":
        if args.list:
            for s in loader.list_scala_tunings():
                print(s)
        elif args.item:
            tuning = loader.load_scala_tuning(args.item)
            print(json.dumps(tuning.to_json(), indent=2))
        else:
            print("Specify --item or --list")
    
    elif args.dataset == "fsd50k":
        df = loader.load_fsd50k()
        print(f"Loaded {len(df)} sounds from FSD50K")
        if args.output and HAS_PANDAS:
            df.to_json(args.output, orient='records', indent=2)
