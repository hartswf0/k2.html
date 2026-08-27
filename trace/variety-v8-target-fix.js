"use strict";
/* TRACE v8.7 · hard prompt targets.
   Fixes carrier conversion: target grammar follows the REQUESTED kind, not the old track kind.
   A focused track remains one-track-only. + PART remains additive. */
(()=>{
  const priorCompose=globalThis.compose;
  const priorVariant=globalThis.v8PlanVariant;
  const priorRenderAll=globalThis.renderAll;
  const priorInit=globalThis.v8Init;

  function isPerc(kind){return kind==="drum"||kind==="percussion"}
  function requestedKind(text){
    const q=String(text||"").toLowerCase();
    if(/violin|viola|cello|contrabass|double bass|strings?|pizzicato/.test(q))return"strings";
    if(/choir|opera|operatic|voice|vocal|singer/.test(q))return"voice";
    if(/sax|clarinet|oboe|bassoon|reed/.test(q))return"reed";
    if(/trumpet|trombone|horn|brass|tuba/.test(q))return"horn";
    if(/flute|shakuhachi|recorder|ocarina|whistle/.test(q))return"flute";
    if(/organ|hammond/.test(q))return"organ";
    if(/rhodes|electric piano|clavinet|\bpiano\b|\bkeys\b/.test(q))return"electric-piano";
    if(/guitar|banjo|fiddle|sitar|koto|shamisen/.test(q))return"guitar";
    if(/marimba|vibraphone|xylophone|kalimba|mbira|mallet/.test(q))return"mallet";
    if(/\bbass\b|upright/.test(q))return"bass";
    if(/percussion|conga|shaker|cowbell|tambourine|clave|agogo|hand drum/.test(q))return"percussion";
    if(/drum kit|\bdrums?\b|kick|snare|hi.?hat|\b808\b/.test(q))return"drum";
    if(/pad|texture|drone|field recording|noise|atmosphere/.test(q))return"texture";
    if(/lead|solo|melody/.test(q))return"lead";
    return null;
  }
  function kindForProgram(program){
    const p=clamp(+program||0,0,127),fam=Math.floor(p/8);
    if(fam===0)return"electric-piano";if(fam===1)return"mallet";if(fam===2)return"organ";if(fam===3)return"guitar";
    if(fam===4)return"bass";if(fam===5)return"strings";if(fam===6)return p>=52?"voice":"strings";if(fam===7)return"horn";
    if(fam===8)return"reed";if(fam===9)return"flute";if(fam===10)return"lead";if(fam===11||fam===12||fam===13||fam===15)return"texture";
    if(fam===14)return"percussion";return"synth";
  }
  function trackById(id){return state.composition?.tracks?.find(t=>+t.id===+id)||null}
  function planScope(plan){
    const s=plan?.trace_target?.scope||plan?.trace_scope||scopeObject();
    const end=beats(state.composition);
    return{kind:s.kind||"song",label:s.label||"SCOPE",start:clamp(+s.start||0,0,end),end:clamp(+s.end||end,0,end)};
  }
  function targetProgram(text,kind,trackId){
    const explicit=typeof v8LocalProgram==="function"?v8LocalProgram(text):null;
    if(explicit!==null)return explicit;
    return isPerc(kind)?null:v8ProgramFor(text,kind,0,trackId);
  }
  function pitchField(scope,excludeTrack,kind){
    const source=state.composition.events.filter(e=>e.type==="note"&&+e.track!==+excludeTrack&&e.start>=scope.start&&e.start<scope.end).map(e=>e.pitch);
    let vals=[...new Set(source)].slice(0,12);
    if(!vals.length)vals=[48,55,60,63,67,70,74];
    const center=kind==="bass"?43:kind==="strings"?67:["lead","flute","mallet","voice"].includes(kind)?72:62;
    return vals.map(p=>{
      let x=p;while(x<center-9)x+=12;while(x>center+12)x-=12;return clamp(x,28,96);
    });
  }
  function evolvingLoop(scope,track,kind,text){
    const q=String(text).toLowerCase(),span=Math.max(.5,scope.end-scope.start),h=v8Hash(text),procedural=/generative|procedural|evolv|mutat|variation|vary|unrepeat|alive/.test(q);
    const step=procedural?(span>=12?.5:.375):(span>=12?.75:.5),count=Math.max(4,Math.min(24,Math.floor((span-.05)/step))),field=pitchField(scope,track,kind),wide=/opera|operatic|dramatic|wide|leap/.test(q);
    const pitches=[];for(let i=0;i<count;i++){
      const idx=(h+i*5+i*i)%field.length;let p=field[idx];
      if(wide&&i%5===3)p+=((h>>>i)%2?12:-12);
      else if(procedural&&i%7===5)p+=((h>>>Math.min(i,28))%3-1)*12;
      pitches.push(clamp(p,32,96));
    }
    const duration=/legato|opera|operatic|sustain/.test(q)?Math.min(step*.92,1.1):Math.min(step*.62,.72);
    return{track:+track,pitches,start:scope.start,step,duration,repeats:pitches.length,velocity:62+(h%24),wave:"sine",reason:procedural?"write one evolving phrase across the captured span instead of repeating a short cell":"write a bounded phrase that belongs to the focused carrier"};
  }
  function peripheralDrum(scope,track,text){
    const h=v8Hash(text),step=[.375,.5,.75,1][h%4],start=scope.start+([0,.125,.25][(h>>>4)%3]),repeats=Math.max(1,Math.min(32,Math.floor((scope.end-start-.12)/step)+1));
    return{track:+track,drum:["kick","snare","shaker","conga","rim","woodblock","tambourine"][h%7],start,step,repeats,velocity:48+(h%36),reason:"give the focused percussion carrier its own bounded pattern"};
  }
  function retargetEffects(plan,track,text){
    const map=new Map();for(const x of plan.effects||[])map.set(x.parameter,{...x,track:+track});
    const q=String(text).toLowerCase();
    if(/electric|electrical|electro|voltage|distort|fuzz/.test(q))map.set("distortion",{track:+track,parameter:"distortion",value:.18,reason:"give the requested electrical edge an audible consequence"});
    if(/opera|operatic|cathedral|large room/.test(q))map.set("reverb",{track:+track,parameter:"reverb",value:.46,reason:"place the focused voice in a larger resonant field"});
    if(/tremolo|shimmer/.test(q))map.set("tremolo",{track:+track,parameter:"tremolo",value:.24,reason:"make amplitude motion audible on this carrier"});
    return[...map.values()].slice(0,V7_EFFECT_LIMIT);
  }
  function trackChangeFor(text,t,kind){
    const program=targetProgram(text,kind,t.id),explicit=typeof v8LocalProgram==="function"?v8LocalProgram(text):null;
    const name=explicit!==null?v8ProgramName(explicit).slice(0,24):(kind!==t.kind?String(kind).replace(/-/g," ").toUpperCase():t.name);
    return{track:+t.id,name,kind,wave:t.wave||"sine",program,reason:kind!==t.kind?`convert CH ${t.id} from ${t.kind} grammar to ${kind} grammar`:"keep the focused carrier identity explicit"};
  }
  function fixTrack(text,plan){
    const p=clone(plan),target=p.trace_target,t=trackById(target?.track);if(!t)return p;
    const scope=planScope(p),explicit=typeof v8LocalProgram==="function"?v8LocalProgram(text):null,rk=requestedKind(text),kind=rk||(explicit!==null?kindForProgram(explicit):t.kind),conversion=kind!==t.kind,perc=isPerc(kind),procedural=/generative|procedural|evolv|mutat|variation|vary|unrepeat|alive/.test(String(text).toLowerCase());
    p.trace_target={...target,name:conversion?trackChangeFor(text,t,kind).name:t.name,kind,scope};p.trace_scope=scope;p.tempo=null;
    p.track_changes=(conversion||explicit!==null)?[trackChangeFor(text,t,kind)]:[];
    p.note_loops=perc?[]:(p.note_loops||[]).slice(0,4).map(x=>({...x,track:+t.id}));
    p.drum_loops=perc?(p.drum_loops||[]).slice(0,8).map(x=>({...x,track:+t.id})):[];
    if(!perc&&(conversion||procedural||!p.note_loops.length))p.note_loops=[evolvingLoop(scope,t.id,kind,text)];
    if(perc&&(conversion||procedural||!p.drum_loops.length))p.drum_loops=[peripheralDrum(scope,t.id,text)];
    p.clear_ranges=(p.note_loops.length||p.drum_loops.length)?[{track:+t.id,start:scope.start,end:scope.end,reason:`replace only CH ${t.id} inside ${scope.label}`}]:[];
    p.effects=retargetEffects(p,t.id,text);
    if(conversion||explicit!==null)p.sample_assignments=[];else p.sample_assignments=(p.sample_assignments||[]).slice(0,1).map(x=>({...x,track:+t.id}));
    p.dimensions=[...new Set([...(p.dimensions||[]),conversion?"orchestration":null,procedural?"form":null].filter(Boolean))].slice(0,7);
    p.summary=`CH ${t.id} · ${conversion?`${t.name} → ${p.trace_target.name}`:(p.summary||"focused rewrite")}`;
    p.interpretation=[`HARD TARGET: only CH ${t.id} inside ${scope.label} may be rewritten.`,conversion?`The carrier changes from ${t.kind} to ${kind}; its old ${isPerc(t.kind)?"percussion":"melodic"} grammar is discarded inside this span.`:`The carrier stays ${t.kind}; only its local behavior changes.`,...(p.interpretation||[])].slice(0,6);
    return validatePlan(p);
  }
  function fixLayer(text,plan){
    const p=clone(plan),target=p.trace_target;if(!target?.track)return p;const scope=planScope(p),change=(p.track_changes||[]).find(x=>+x.track===+target.track),kind=requestedKind(text)||change?.kind||target.kind||"synth",perc=isPerc(kind),procedural=/generative|procedural|evolv|mutat|variation|vary|unrepeat|alive/.test(String(text).toLowerCase());
    p.trace_target={...target,kind,scope};p.trace_scope=scope;p.tempo=null;p.clear_ranges=[];
    if(change&&requestedKind(text)){const program=targetProgram(text,kind,target.track);Object.assign(change,{kind,program,name:Number.isInteger(program)?v8ProgramName(program).slice(0,24):String(kind).toUpperCase()})}
    p.note_loops=perc?[]:(p.note_loops||[]).slice(0,4).map(x=>({...x,track:+target.track}));
    p.drum_loops=perc?(p.drum_loops||[]).slice(0,8).map(x=>({...x,track:+target.track})):[];
    if(!perc&&(procedural||!p.note_loops.length))p.note_loops=[evolvingLoop(scope,target.track,kind,text)];
    if(perc&&!p.drum_loops.length)p.drum_loops=[peripheralDrum(scope,target.track,text)];
    p.effects=retargetEffects(p,target.track,text);if(requestedKind(text))p.sample_assignments=[];
    p.summary=`ADD CH ${target.track} · ${change?.name||target.name||kind}`;
    p.interpretation=[`HARD TARGET: add one carrier inside ${scope.label}. Existing channels are not cleared.`,...(p.interpretation||[])].slice(0,6);
    return validatePlan(p);
  }
  function fix(text,plan){
    if(!plan?.trace_target||plan.trace_target.mode==="all")return plan;
    return plan.trace_target.mode==="track"?fixTrack(text,plan):fixLayer(text,plan);
  }

  globalThis.compose=async function(text){return fix(text,await priorCompose(text))};
  globalThis.v8PlanVariant=function(text,plan,degree){return fix(text,priorVariant(text,plan,degree))};

  function syncAffordance(){
    const L=globalThis.TRACE_LAYER,send=byId("sendBtn"),input=byId("prompt");if(!L||!send||!input)return;
    if(L.mode==="track"&&L.trackId){const t=trackById(L.trackId);send.textContent=`SEND CH ${L.trackId}`;input.placeholder=`Tell CH ${L.trackId}${t?` · ${t.name}`:""} what to become or how to change…`;}
    else if(L.mode==="layer"){send.textContent="ADD PART";input.placeholder="Describe one new part to layer into the selected span…";}
    else{send.textContent="SEND";input.placeholder="Make, ask why, ask otherwise, or describe what you hear…";}
  }
  globalThis.renderAll=function(){priorRenderAll();syncAffordance()};
  globalThis.v8Init=function(){priorInit();syncAffordance();document.documentElement.dataset.traceTargetFix="8.7"};
})();
