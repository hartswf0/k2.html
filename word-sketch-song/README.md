# WORD / SKETCH / SONG

Static recursive music instrument for GitHub Pages.

## Surface

- WORD / SKETCH
- SONG / SPECTRUM
- CODE / FLOW
- GUIDE / HISTORY

Every MAKE or LOOP creates a provenance branch. Any panel can be sent to the GUIDE. Images can be dropped or pasted into the GUIDE.

## Export

- MP3 (WAV fallback if the encoder CDN is unavailable)
- MIDI
- standalone song HTML with provenance
- TRACE JSON with pipeline, history and graph

The GitHub Pages frontend calls a narrow Supabase Edge Function, which forwards only the allowed OpenAI endpoints. The entered OpenAI API key is passed per request and is not stored by the instrument. The Supabase publishable key in the frontend is intentionally public.
