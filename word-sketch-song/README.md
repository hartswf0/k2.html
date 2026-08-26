# WORD / SKETCH / SONG

A static, no-scroll AI music instrument.

## Run

No Flask server is required. Host this folder on any static HTTPS host (GitHub Pages is ideal), or serve it locally:

```bash
python3 -m http.server 8080
```

Then open `http://127.0.0.1:8080`.

Enter your own OpenAI API key on the first screen and press **CHECK**. The key is kept only in the current page's JavaScript memory; it is not written to files or committed to the repository.

## Surface

- WORD / SKETCH: source material
- SONG / SPECTRUM: audible and visual consequence
- CODE / CHANGE: causal trace and critique
- MAKE creates the first A1–B–A2 song
- LOOP regenerates from current words, sketch, spectrum marks, critique, and previous agent state
- Browser playback, MIDI download, JSON download, and generated `music21` Python are included

For a production deployment with a shared API key, put the OpenAI calls behind a server-side or serverless proxy instead of exposing a secret in the browser.
