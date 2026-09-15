# Pond: theory of the program

## 1. <Initial Interpretation>

The real activity is composing a timed passage by listening, describing, revising, and comparing. A <pond> is a <composition workspace>. It is not a lyric generator that happens to contain a music application.

<composition> [contains] {tracks, timed notes, timed lyric lines}.
<musician> [selects] <passage>.
<direction> [constrains] <proposal>.
<candidate> [auditions beside] <original>.
<acceptance> [commits] <revision>.

## 2. <Theory Skeleton>

<entities> := {composition, track, note, bar, lyric line, passage, selection, lock, candidate, revision, writing draft}.

[operations] := {[select], [write], [lock], [propose], [validate], [audition], [compare], [commit], [discard], [undo], [resize], [save], [open]}.

<editing>, <requesting>, and <candidate-ready> describe proposal state. <playing-original>, <playing-candidate>, and <stopped> describe the independent audio transport.

<conditions>:
- A selection contains complete, existing bars.
- Every note fits wholly within a four-beat bar.
- A proposal contains the existing track IDs and selected bar count.
- [commit] requires a validated candidate from the current revision and selection.
- Generated changes must respect the requested music/lyrics scope.
- [generate] requires a key, a configured model, and a direction.
- User-triggered playback resumes the audio context.

<invariants>:
1. Proposing cannot mutate the authoritative composition.
2. Material outside the selected range is preserved exactly.
3. Locked tracks and locked lyric lines are preserved by candidate validation.
4. Music-only proposals preserve lyrics; lyrics-only proposals preserve music.
5. Invalid or stale responses cannot commit.
6. Undo restores the prior composition.
7. One scheduler owns audio; switching auditions stops prior nodes.
8. Project output excludes the application's API credentials.

## 3. <Assumption Ledger>

- <safe>: One timeline is the shared coordinate system for music and words.
- <safe>: User acceptance is explicit.
- <safe>: Word changes should be inspectable before use.
- <uncertain>: Four beats per bar and one lyric line per bar are sufficient for this version.
- <uncertain>: Four synthesized instruments adequately expose the interaction.
- <uncertain>: A 4–32-bar range is useful for early composition.
- <requires-user-decision>: None blocks this implementation.

Local variations transpose pitched notes and vary velocity. They are deliberately simple transformations, not semantic interpretations of the direction. Semantic changes use the configured model.

## 4. <Operational Description>

1. <starter score> [opens as] <eight-bar composition>.
2. <musician> [selects] <range>, using bar buttons, shift-click, or start/end inputs.
3. <musician> [writes] <lyric draft>, one line per bar. Drafts remain separate until [Apply lyrics].
4. <musician> [locks] {tracks, selected lyric lines}.
5. {selected score, lyric lines, locks, scope, direction} [transforms into] <request>.
6. <response> [passes through] <local validator>.
7. <valid response> [becomes] <candidate>; <invalid response> [becomes] <visible rejection>.
8. <original> and <candidate> [play through] <one transport>.
9. <Use candidate> [commits] <selected replacement>.
10. <Undo> [restores] <previous composition>.
11. <Score> exposes editable JSON that passes the same candidate validator.
12. <Project> [saves or opens] <portable composition plus current drafts>.

Selection or composition changes cancel in-flight requests. The controller checks a request identity; the model checks revision and selection. These checks serve different purposes.

## 5. <Failure Description>

- Invalid JSON: reject it, preserve composition.
- Missing/duplicate track identity: reject it.
- Out-of-range pitch, duration, velocity or beat: reject it.
- Changed protected material: reject it.
- Network or model error: display the error and preserve work.
- Timeout: abort after 90 seconds.
- Selection/revision changes during generation: cancel and reject stale results.
- Locked lyric editing: block [Apply] and retain the draft.
- Invalid project: validate before replacing the composition.
- Storage unavailable: retain in-memory work and direct the user to project export.
- Audio resume failure: stop playback and display the error.
- Oversized project: reject files above 3 MB.

Unapplied lyric drafts are preserved by bar range, including across music changes. Undo restores the composition; it does not rewind every keystroke. History is in-memory and capped at thirty composition revisions. Project files and autosave preserve the current score and writing drafts, not the full undo stack.

## 6. <Change Test>

“What would change if the requirements changed?”

### More bars

Extending from eight to thirty-two bars appends empty musical bars and lyric slots. Selection, validation, comparison and commit still operate on the same passage representation. A test edits the final two bars after extension.

### A different tempo during generation

Tempo changes advance the composition revision. An earlier proposal cannot commit against the new revision. This is tested.

### Recorded vocals or imported samples

Introduce <audio clip> with timing, buffer/source identity and duration. Extend the validator and audio renderer. Preserve the candidate/original distinction and selected-range replacement. Do not pretend oscillators already implement this.

### Another model/provider

Replace the request adapter. Keep validation local and provider-independent. The current model configuration is a free-text model ID. Availability and output quality must be checked against the user's account.

## 7. <Implementation Plan>

- pond-core.js: DOM-free composition model, passage extraction, validation, replacement, sessions and revision history.
- pond-app.js: timeline, writing drafts, controls, generation adapter, audio scheduler, persistence.
- pond.html: one shared workspace and dialogs.
- pond.css: timeline-centered desktop layout; scrollable panels on mobile.
- pond-tests.cjs: protection and modification tests.
- pond-controller-tests.cjs: controller tests with a stub DOM and mocked API.

No framework, backend, or build step is required. Serve these files together over HTTP(S) and open pond.html.

## 8. <Program Text>

The executable program is in the files above.

Run the reproducible tests with:

    node pond-tests.cjs
    node pond-controller-tests.cjs

This version is a separate entry point. The earlier polliwog-earsketch.html prototype is not the Pond container.

## 9. <Theory-Code Mapping>

| Theory | Code |
|---|---|
| <types> | Runtime-validated shapes for composition, tracks, notes, lyrics, selections and proposals |
| [extract passage] | PondCore.passage |
| [protect conditions] | PondCore.proposal and PondCore.validate |
| [apply local replacement] | PondCore.apply |
| <states> and <revision> | PondCore.session closure |
| [commit] / [undo] | Session methods |
| <writing draft> | Controller drafts indexed by selected bar range |
| [audition] | Web Audio scheduler with cancellation token and shared master gain |
| <configuration> | Tempo, length, scope, key and model controls |
| <tests> | Pure core assertions and mocked controller transitions |
| <comments> | Source comments identify authoritative composition and separate candidate |
| <classes> | Production code uses functions/closures; no domain classes are necessary |

Validation performed during construction: 15 core checks and 10 controller checks passed in an independent JavaScript runtime. Both JavaScript files passed syntax compilation. Controller checks used a stub DOM and mocked API.

The execution workspace stopped responding. Real-browser layout, audible sound quality, and live API calls have NOT been verified for this rebuild. Earlier prototype browser results must not be treated as verification of Pond.

## 10. <Residual Human Theory>

The program protects boundaries, not musical quality. A maintainer must understand why the passage is the unit of change, why a candidate is not committed work, and why locks are validated locally.

Pitch steps and oscillator tones are an instrumental sketch. They do not represent full orchestration, EarSketch sample integration, recorded vocals, sung lyrics, arbitrary meter, or phrase timing below a bar. Cultural appropriateness, singability, expressive timing, and whether a revision improves the piece remain human judgments.

When expanding the program, preserve the distinction between a lyric draft, a validated candidate, and the committed score. Do not silently merge them to simplify the interface.
