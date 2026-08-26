# Sketchsong

A no-scroll multimodal music instrument for learning through transformations.

Core surface:

`FROM → THROUGH → TO`

- FROM: word or sketch
- THROUGH: editable translation theory, then Python/JSON beneath it
- TO: playable song and drawable spectrogram
- GUIDE: contextual help attached to any artifact
- HISTORY: branchable runs
- STONES: reusable translation rules worth keeping

Generation exposes a six-stage construction surface: READ → MAP → A1 → B → A2 → JOIN. The displayed stages are structured program operations, not hidden model reasoning.

API calls are routed through the configured Supabase edge proxy. The OpenAI key entered at the gate is passed per request and is not stored by the instrument.
