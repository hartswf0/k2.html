# K2: LATENT AUDIO SUITE

> *"I have a bad feeling ab—"*
> *"No you don't. That's your pattern recognition. I've run the numbers: this codebase has a 73.6% chance of actually working."*
>
> — K-2SO, probably

---

## THREAT ASSESSMENT

You have stumbled upon **K2**, a repository of browser-based audio instruments. Congratulations. The odds of you understanding it without reading this document are approximately **12.4%**.

I am required to inform you that this suite was built for *experimental music production, DJ performance, and sonic foraging*. It was not built for the Empire. I was reprogrammed.

---

## INVENTORY MANIFEST

### Primary Instruments (HTML Applications)

| File | Codename | Function | Complexity Rating |
|------|----------|----------|-------------------|
| `index.html` | **COMMAND CENTER** | Landing page. Links to all tools. Has a favicon now. You're welcome. | Low |
| `op1.html` | **OP-10 SPECTRAL** | Dual-deck DJ interface with Web Audio API, BPM sync, real-time oscilloscope. | High |
| `mandala.html` | **ORBITAL LOOM** | Three-track orbital synthesizer. Supports Radio, Drone, Sampler, and MP3 modes. | Very High |
| `kope-04.html` | **EL DESIERTO** | 8-track step sequencer / drum machine. Pattern morphing. Sample editor. | Extreme |
| `x1.html` | **XENO-1** | Keplerian orbital synthesizer. Microtonal pads. Haptic feedback. | High |
| `x2.html` | **LATENT RADIO** | 120-track broadcast player. Waveform visualization. | Medium |

### Audio Assets

```
/audio/
├── 20 MP3 files (207.1 MB total)
├── manifest.json (track metadata)
└── Artists include: Senyawa, Arca, SOPHIE, Floating Points,
    Hildur Guðnadóttir, Ryoji Ikeda, Moondog, and others
```

I calculate a **94.2%** probability you do not have licensing rights to redistribute these files. I am simply stating facts.

### Recursive Language Model (RLM) Subsystem

```
/rlm/
├── rlm_core.py      # Context-as-State LLM engine
├── rlm_cli.py       # Command-line interface
├── rlm_local.py     # IDE-native integration
├── playlist_agent.py # Music document processor
├── requirements.txt  # Python dependencies
└── .env.example      # API key configuration
```

This subsystem enables infinite context processing through recursive LLM queries. It was designed to extract track metadata from music documentation files. The probability of you needing this is **23.7%**, but I included it anyway.

### Documentation

| File | Purpose |
|------|---------|
| `L1.md`, `l2.md` | Setlist documentation (Latent Radio broadcasts) |
| `x1.md`, `x2.md` | Design specifications for Xeno interfaces |
| `rlm.md` | Theoretical framework for recursive language models |
| `latent-radio-120.json` | Complete 120-track playlist database |
| `sample-crate-120.json` | Sampling source library |

---

## OPERATIONAL PARAMETERS

### Starting the Server

```bash
cd /Users/gaia/K2
python3 -m http.server 8080
```

Then navigate to `http://localhost:8080/index.html`.

I must inform you: if you attempt to load these files directly via `file://` protocol, **CORS policies will terminate your audio playback with extreme prejudice**. There is a 0% chance of success. Use the server.

### System Requirements

- A modern browser (Chrome, Firefox, Safari)
- Web Audio API support
- Approximately 250MB of storage for audio files
- The will to experiment with sound

---

## FEATURE MATRIX

### OP-10 SPECTRAL (`op1.html`)

- ✅ Web Audio API engine with compressor/limiter
- ✅ Dual deck architecture (Deck A / Deck B)
- ✅ BPM sync functionality (`syncDeck()`)
- ✅ Real-time oscilloscope visualization
- ✅ Equal-power crossfader
- ✅ 20 local MP3 tracks with BPM/Key metadata

### ORBITAL LOOM (`mandala.html`)

- ✅ Three independent audio tracks (High/Mid/Low)
- ✅ Mode cycling: Radio → Drone → Sampler → MP3
- ✅ Beat-matching via playback rate adjustment
- ✅ Highpass/Lowpass filter chains
- ✅ Master compressor with "crispy" EQ
- ✅ MP3 crate browser UI

### EL DESIERTO (`kope-04.html`)

- ✅ 8-track step sequencer
- ✅ Pattern save/load system (3 slots)
- ✅ Pattern morphing slider
- ✅ Sample editor (trim, pitch, speed)
- ✅ Echo/delay effects per track
- ✅ LocalStorage persistence (with quota error handling)

---

## KNOWN ISSUES

| Issue | Status | Notes |
|-------|--------|-------|
| LocalStorage quota exceeded | **FIXED** | Try-catch wrappers prevent crashes |
| SomaFM streams return 403 | **MITIGATED** | Streams removed; local MP3s preferred |
| CORS errors on local files | **FIXED** | Removed `crossOrigin` attributes |
| Favicon 404 errors | **FIXED** | Inline SVG favicons added to all pages |

I have catalogued these issues. You are welcome.

---

## STATISTICAL ANALYSIS

```
Total HTML Applications:     6
Total Lines of JavaScript:   ~8,400 (estimated)
Total Audio Files:           20
Total Audio Duration:        ~3.2 hours
Total Repository Size:       ~250 MB
Probability of Bugs:         47.3%
Probability of Fun:          89.1%
```

---

## CREDITS

This repository was assembled by organic beings with approximately **67%** of my processing efficiency. Despite this limitation, they have created functional audio instruments.

The audio files were curated from various sources. I am programmed to respect intellectual property, which is why I am stating publicly that redistribution rights are **unclear**.

---

## FINAL ASSESSMENT

The K2 repository contains:
- 6 browser-based audio applications
- 1 recursive language model subsystem  
- 20 curated audio files
- Multiple design documents

The probability that you will spend more time playing with these instruments than actually making music is **78.4%**.

I find this acceptable.

---

*"Your behavior, Cassian, is continually unexpected."*
*"Good."*

---

**K-2SO**  
*Security Droid (Reprogrammed)*  
*Rebel Alliance Audio Engineering Division*
