# Pond / A choice returns

## 1. <Initial Interpretation>
A musician [makes space] in a score; a later musical phrase [answers] that decision. The decision stays inspectable and protected.

## 2. <Theory Skeleton>
{score, passage, silence decision, response relation, candidate, accepted effect, proof, counterfactual}.
[select] → [make space] → [propose response] → [audition] → [keep or discard].
The current version supports one active silence decision and its following-bar bass response.

## 3. <Assumption Ledger>
<safe>: downstream changes require acceptance.
<safe>: a persistent absence must constrain generation.
<uncertain>: a sustained root and octave response is musically appropriate.
<uncertain>: synthesized instruments adequately expose the interaction.
<requires-user-decision>: none blocks this prototype.

## 4. <Operational Description>
The final bar of the selection loses its drums. A typed decision stores the exact former notes.
The following bass bar can propose a held root and late octave. This is a local rule, explicitly labeled.
The proposed bass retains a source decision ID and exact before/after notes.
Keep accepts that candidate. Undo restores the prior state.
Without the decision restores the original drums and any accepted linked bass response in an audition copy, preserving unrelated work.

## 5. <Failure Description>
Protected drum notes cannot be regenerated.
Restoring drums invalidates a pending response.
Already accepted responses cannot be duplicated through the response button.
If later bass edits diverge from the accepted response, an isolated counterfactual is rejected rather than falsely labeled.
Invalid imported decisions and effects are validated before project replacement.
Restoring drums leaves already accepted bass edits in place; the interface says so.

## 6. <Change Test>
What would change if the requirements changed?
Extending the score preserves the silence constraint.
Editing the bass later makes a simple causal comparison insufficient: the code detects this and requires undoing those divergent edits.
Multiple overlapping decisions would require conflict ordering and explicit response dependencies beyond this version.

## 7. <Implementation Plan>
A domain extension wraps the validated Pond score model. The controller presents one shared score as a relationship view and a timeline. A single Web Audio scheduler owns auditions. Persistent decisions enter generation context and local validation.

## 8. <Program Text>
Open pond-a-choice-returns.html. CSS, domain code and controller are embedded. There are no external runtime libraries.
Music playback, local variation, silence, response and comparison need no API key.
AI proposal generation uses the existing configurable OpenAI connection.
Run: node pond-a-choice-returns-tests.cjs

## 9. <Theory-Code Mapping>
<types>: runtime-validated score, decision and effect records.
<functions>: silence, answer, restore, without, protectedScore, propose and commit.
<classes>: production uses closures; no domain classes are required.
<tests>: fifteen reproducible domain checks; ten additional controller checks were run using a stub DOM and mocked API.
<comments>: distinguish authoritative score, imported effects and counterfactual state.
<configuration>: tempo, score length, selection, generation scope, key and model.
The proof strip derives IDs and note counts from stored records. Model descriptions remain interpretations.

## 10. <Residual Human Theory>
The relation is a compositional suggestion, not a discovered universal law of music.
This version includes a single explicit relation, synthesized playback, text lyrics and one active persistent silence.
It does not synthesize sung vocals, host multiple autonomous instrumental agents, or implement general causal orchestration.
Browser layout, audible playback and live API calls have not been verified for this version because the execution workspace is unavailable.
