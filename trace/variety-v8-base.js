"use strict";
/* Explicit compatibility seam. v8 never guesses which previous function it is wrapping. */
globalThis.TRACE_V7_BASE=Object.freeze({
  boot,
  compose,
  applyPlan,
  sourcePatch,
  context,
  renderAll,
  renderTrace,
  renderMessages,
  previewMessage,
  applyMessage,
  sendPrompt,
  lineTheory,
  compileSource,
  parseSource,
  play
});
