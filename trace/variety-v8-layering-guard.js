"use strict";
/* TRACE v8.6 target guard.
   Focused-track prompts keep the carrier identity unless the user explicitly names another voice. */
(()=>{
  const priorCompose=globalThis.compose;
  const priorVariant=globalThis.v8PlanVariant;

  function requestedKind(text){
    const q=String(text).toLowerCase();
    if(/drum kit|\bdrums?\b|kick|snare|hi.?hat/.test(q))return"drum";
    if(/percussion|conga|shaker|cowbell|tambourine|clave|agogo|hand drum/.test(q))return"percussion";
    if(/bass|upright/.test(q))return"bass";
    if(/organ|hammond/.test(q))return"organ";
    if(/guitar|banjo|fiddle|sitar|koto|shamisen/.test(q))return"guitar";
    if(/sax|clarinet|oboe|bassoon|reed/.test(q))return"reed";
    if(/trumpet|trombone|horn|brass|tuba/.test(q))return"horn";
    if(/flute|shakuhachi|recorder|ocarina|whistle/.test(q))return"flute";
    if(/strings|violin|viola|cello|pizzicato/.test(q))return"strings";
    if(/rhodes|electric piano|clavinet|piano|keys/.test(q))return"electric-piano";
    if(/marimba|vibraphone|xylophone|kalimba|mbira|mallet/.test(q))return"mallet";
    if(/choir|voice|vocal/.test(q))return"voice";
    if(/pad|texture|drone|field recording|noise|atmosphere/.test(q))return"texture";
    if(/lead|solo|melody/.test(q))return"lead";
    return null;
  }
  function kindForProgram(program){
    const p=clamp(+program||0,0,127),fam=Math.floor(p/8);
    return fam===0?"electric-piano":fam===1?"mallet":fam===2?"organ":fam===3?"guitar":fam===4?"bass":fam===5?"strings":fam===6?(p>=52?"voice":"strings"):fam===7?"horn":fam===8?"reed":fam===9?"flute":fam===10?"lead":[11,12,13,15].includes(fam)?"texture":fam===14?"percussion":"synth";
  }
  function normalizeFocusedVoice(text,plan){
    const p=clone(plan),target=p?.trace_target;if(target?.mode!=="track")return p;
    const track=state.composition.tracks.find(t=>+t.id===+target.track);if(!track)return p;
    const rk=requestedKind(text),ep=typeof v8LocalProgram==="function"?v8LocalProgram(text):null;
    if(!rk&&ep===null){p.track_changes=[];return validatePlan(p)}
    const program=ep!==null?ep:v8ProgramFor(text,rk||track.kind,1,+track.id),kind=rk||kindForProgram(program),existing=p.track_changes?.find(x=>+x.track===+track.id),name=(existing?.name&&existing.name!=="NEW PART")?existing.name:v8ProgramName(program).slice(0,24);
    p.track_changes=[{track:+track.id,name,kind,wave:existing?.wave||track.wave||"sine",program,reason:existing?.reason||"the focused prompt explicitly changes this carrier's voice"}];
    p.trace_target={...target,name,kind};
    return validatePlan(p);
  }
  globalThis.compose=async function(text){return normalizeFocusedVoice(text,await priorCompose(text))};
  globalThis.v8PlanVariant=function(text,plan,degree){return normalizeFocusedVoice(text,priorVariant(text,plan,degree))};
})();
