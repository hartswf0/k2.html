"use strict";
/* TRACE v8.5 · low-road composition.
   A prompt always has an explicit carrier target: ALL, THIS TRACK, or NEW PART.
   Time scope and carrier scope compose. Existing music is never cleared by NEW PART. */
(()=>{
  const priorCompose=globalThis.compose;
  const priorVariant=globalThis.v8PlanVariant;
  const priorContext=globalThis.context;
  const priorRenderAll=globalThis.renderAll;
  const priorRenderTimeline=globalThis.renderTimeline;
  const priorRenderMessages=globalThis.renderMessages;
  const priorApplyMessage=globalThis.applyMessage;
  const priorInit=globalThis.v8Init;

  const layerState=globalThis.TRACE_LAYER={mode:"all",trackId:null,lastTrackId:null};

  function trackById(id,c=state.composition){return c?.tracks?.find(t=>+t.id===+id)||null}
  function activeOn(t,c=state.composition){return c.events.some(e=>+e.track===+t.id)}
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
    if(fam===0)return"electric-piano";
    if(fam===1)return"mallet";
    if(fam===2)return"organ";
    if(fam===3)return"guitar";
    if(fam===4)return"bass";
    if(fam===5)return"strings";
    if(fam===6)return p>=52?"voice":"strings";
    if(fam===7)return"horn";
    if(fam===8)return"reed";
    if(fam===9)return"flute";
    if(fam===10)return"lead";
    if(fam===11||fam===12||fam===13||fam===15)return"texture";
    if(fam===14)return"percussion";
    return"synth";
  }
  function scopeFromPlan(plan){
    const live=scopeObject(),s=plan?.trace_scope||live;
    return{kind:s.kind||live.kind,label:s.label||live.label,start:+s.start,end:+s.end};
  }
  function targetSnapshot(){
    const s=scopeObject();
    if(layerState.mode==="track"&&trackById(layerState.trackId)){
      const t=trackById(layerState.trackId);
      return{mode:"track",track:+t.id,name:t.name,kind:t.kind,scope:{kind:s.kind,label:s.label,start:s.start,end:s.end}};
    }
    if(layerState.mode==="layer")return{mode:"layer",track:null,name:"NEW PART",kind:null,scope:{kind:s.kind,label:s.label,start:s.start,end:s.end}};
    return{mode:"all",track:null,name:"ALL",kind:null,scope:{kind:s.kind,label:s.label,start:s.start,end:s.end}};
  }
  function targetLabel(target){
    if(!target||target.mode==="all")return"ALL TRACKS";
    if(target.mode==="layer")return"NEW PART";
    return`CH ${target.track} · ${target.name||"TRACK"}`;
  }
  function explicitProgram(text){try{return typeof v8LocalProgram==="function"?v8LocalProgram(text):null}catch{return null}}
  function pickPlanChange(plan,text,kind){
    const ep=explicitProgram(text);
    if(ep!==null)return{name:v8ProgramName(ep).slice(0,24),kind:kindForProgram(ep),wave:"sine",program:ep,reason:"the prompt explicitly names this voice"};
    const loopTrack=plan.note_loops?.[0]?.track??plan.drum_loops?.[0]?.track;
    const found=(kind&&plan.track_changes?.find(x=>x.kind===kind))||plan.track_changes?.find(x=>+x.track===+loopTrack)||plan.track_changes?.[0];
    if(found)return{...found};
    return{name:"NEW PART",kind:kind||"synth",wave:"sine",program:null,reason:"a separate carrier for this added part"};
  }
  function freeTrack(kind){
    const c=state.composition,inactive=c.tracks.filter(t=>!activeOn(t,c));
    const t=(kind&&inactive.find(x=>x.kind===kind))||inactive[0];
    if(t)return t;
    const used=new Set(c.tracks.map(t=>+t.id));
    const id=Array.from({length:V7_TRACK_LIMIT},(_,i)=>i+1).find(x=>!used.has(x));
    if(!id)throw Error("Every TRACE channel is active. Fork or clear a carrier before adding another part.");
    return{id,name:`CH ${id}`,kind:kind||"synth",wave:"sine"};
  }
  function dedupeEffects(arr,track){
    const map=new Map();
    for(const x of arr||[])map.set(x.parameter,{...x,track:+track});
    return[...map.values()].slice(0,V7_EFFECT_LIMIT);
  }
  function songPitchField(scope,kind){
    const vals=state.composition.events.filter(e=>e.type==="note"&&e.start>=scope.start&&e.start<scope.end).map(e=>e.pitch);
    const uniq=[...new Set(vals)].slice(0,8);
    let p=uniq.length?uniq:[60,63,67,70];
    const shift=kind==="bass"?-12:["flute","lead","mallet"].includes(kind)?12:0;
    return p.slice(0,6).map(x=>clamp(x+shift,24,100));
  }
  function fallbackLayerLoop(scope,track,kind,text){
    const span=Math.max(.25,scope.end-scope.start),pitches=songPitchField(scope,kind),h=v8Hash(text),step=[.75,1,1.5,2][h%4],start=scope.start+([0,.25,.5][(h>>>3)%3]);
    const repeats=Math.max(1,Math.min(12,Math.floor((scope.end-start-.1)/step)+1));
    return{track:+track,pitches,start,step,duration:Math.min(step*.58,.8),repeats,velocity:58+(h%24),wave:"sine",reason:"derive a small answering layer from the song's existing pitch field"};
  }
  function fallbackPercLoop(scope,track,text){
    const h=v8Hash(text),step=[.5,.75,1,1.5][h%4],start=scope.start+([.125,.25,.375][(h>>>4)%3]),repeats=Math.max(1,Math.min(24,Math.floor((scope.end-start-.1)/step)+1));
    return{track:+track,drum:["shaker","conga","rim","woodblock","tambourine"][h%5],start,step,repeats,velocity:42+(h%28),reason:"add a peripheral cycle without replacing the existing kit"};
  }
  function remapLoops(list,track,max){return(list||[]).slice(0,max).map(x=>({...x,track:+track}))}

  function shapeTrack(text,plan,target){
    const p=clone(plan),t=trackById(target.track),scope=scopeFromPlan(p);if(!t)return p;
    const percussion=["drum","percussion"].includes(t.kind),ep=explicitProgram(text),rk=requestedKind(text);
    p.trace_target={...target,scope};p.tempo=null;
    p.track_changes=[];
    if(ep!==null||rk){
      const pc=pickPlanChange(p,text,rk||t.kind),program=ep!==null?ep:(Number.isInteger(pc.program)?pc.program:v8TrackProgram(t));
      p.track_changes=[{track:+t.id,name:pc.name||t.name,kind:pc.kind||rk||t.kind,wave:pc.wave||t.wave||"sine",program,reason:pc.reason||"change only the focused carrier"}];
    }
    p.note_loops=percussion?[]:remapLoops(p.note_loops,t.id,4);
    p.drum_loops=percussion?remapLoops(p.drum_loops,t.id,8):[];
    const structural=p.note_loops.length||p.drum_loops.length;
    p.clear_ranges=structural?[{track:+t.id,start:scope.start,end:scope.end,reason:`replace only CH ${t.id} inside ${scope.label}`}]:[];
    p.effects=dedupeEffects(p.effects,t.id);
    p.sample_assignments=(p.sample_assignments||[]).slice(0,1).map(x=>({...x,track:+t.id}));
    p.summary=`${targetLabel(p.trace_target)} · ${p.summary||"focused change"}`;
    p.interpretation=[`Only ${targetLabel(p.trace_target)} may change. Every other carrier is held.`,...(p.interpretation||[])].slice(0,6);
    return validatePlan(p);
  }
  function shapeLayer(text,plan,target){
    const p=clone(plan),scope=scopeFromPlan(p),rk=requestedKind(text),pc=pickPlanChange(p,text,rk),kind=rk||pc.kind||"synth",t=freeTrack(kind),id=+t.id,percussion=["drum","percussion"].includes(kind),ep=explicitProgram(text);
    const program=ep!==null?ep:(Number.isInteger(pc.program)?pc.program:(percussion?null:v8ProgramFor(text,kind,1,id)));
    p.trace_target={mode:"layer",track:id,name:pc.name||t.name||`CH ${id}`,kind,scope};p.tempo=null;p.clear_ranges=[];
    p.track_changes=[{track:id,name:(pc.name||t.name||`CH ${id}`).slice(0,24),kind,wave:pc.wave||t.wave||"sine",program,reason:pc.reason||"give the new part its own carrier"}];
    p.note_loops=percussion?[]:remapLoops(p.note_loops,id,4);
    p.drum_loops=percussion?remapLoops(p.drum_loops,id,8):[];
    if(!percussion&&!p.note_loops.length)p.note_loops=[fallbackLayerLoop(scope,id,kind,text)];
    if(percussion&&!p.drum_loops.length)p.drum_loops=[fallbackPercLoop(scope,id,text)];
    p.effects=dedupeEffects(p.effects,id);
    p.sample_assignments=(p.sample_assignments||[]).slice(0,1).map(x=>({...x,track:id}));
    p.summary=`ADD CH ${id} · ${p.summary||pc.name||"new part"}`;
    p.interpretation=[`This is additive. CH ${id} enters inside ${scope.label}; existing carriers are not cleared.`,...(p.interpretation||[])].slice(0,6);
    return validatePlan(p);
  }
  function shapePlan(text,plan,target){
    if(!target||target.mode==="all"){const p=clone(plan);p.trace_target={...(target||targetSnapshot()),scope:scopeFromPlan(p)};return p}
    return target.mode==="track"?shapeTrack(text,plan,target):shapeLayer(text,plan,target);
  }

  globalThis.context=function(){
    const c=priorContext(),t=targetSnapshot();c.prompt_target=t;
    if(t.mode==="track")c.events=(c.events||[]).filter(e=>+e.t===+t.track);
    c.low_road="Build by reversible local parts. ALL may rewrite broadly; THIS TRACK may touch one carrier; NEW PART may add one carrier and may not clear existing music.";
    return c;
  };
  globalThis.compose=async function(text){const target=targetSnapshot();return shapePlan(text,await priorCompose(text),target)};
  globalThis.v8PlanVariant=function(text,plan,degree){const target=plan?.trace_target||targetSnapshot();return shapePlan(text,priorVariant(text,plan,degree),target)};

  function setLayerMode(mode,trackId=null){
    if(mode==="track"){
      const t=trackById(trackId);if(!t)return;layerState.mode="track";layerState.trackId=+t.id;layerState.lastTrackId=+t.id;
    }else{layerState.mode=mode;layerState.trackId=null}
    syncLayerUI();renderTimeline();renderMessages();
    if(innerWidth<=820&&mode!=="all")setMobile("chat");
    log("prompt_target",targetSnapshot());
  }
  function installLayerUI(){
    if(byId("traceLayerBar"))return;
    const composer=document.querySelector(".composer"),scope=document.querySelector(".composer .scopeChip");if(!composer||!scope)return;
    const bar=document.createElement("div");bar.id="traceLayerBar";bar.className="traceLayerBar";bar.innerHTML=`<button id="traceTargetAll">ALL</button><button id="traceTargetTrack">TRACK</button><button id="traceTargetNew">+ PART</button>`;scope.insertAdjacentElement("afterend",bar);
    byId("traceTargetAll").onclick=()=>setLayerMode("all");
    byId("traceTargetTrack").onclick=()=>{if(layerState.lastTrackId)setLayerMode("track",layerState.lastTrackId);else showCoach("PICK A TRACK","Tap a channel name in WORLD. The next prompt will be bound to that carrier.")};
    byId("traceTargetNew").onclick=()=>setLayerMode("layer");
    if(!byId("traceLayerStyle")){
      const s=document.createElement("style");s.id="traceLayerStyle";s.textContent=`.traceLayerBar{display:grid;grid-template-columns:.72fr 1.2fr .9fr;min-height:42px;border-bottom:1px solid #000;background:#fff}.traceLayerBar button{min-width:0;border:0;border-right:1px solid #000;background:#fff;padding:0 8px;font:900 9px/1 Arial;letter-spacing:.03em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.traceLayerBar button:last-child{border-right:0}.traceLayerBar button.active{background:#000;color:#fff}.traceLayerBar #traceTargetNew.active{background:#ef2200;color:#fff}.v8VoiceLabel.traceFocused{background:#000!important;color:#fff!important;outline:2px solid #ef2200;outline-offset:-2px}.layerPlanTarget{margin:7px 0 5px;padding:7px 8px;border:1px solid #000;font:900 8px/1 Arial;letter-spacing:.05em;background:#fff;color:#000}.layerPlanTarget.add{background:#ef2200;color:#fff;border-color:#ef2200}@media(max-width:820px){.traceLayerBar{position:sticky;top:0;z-index:4;min-height:48px}.traceLayerBar button{min-height:48px;font-size:10px}.v8VoiceLabel{touch-action:manipulation}}`;document.head.appendChild(s);
    }
    syncLayerUI();
  }
  function syncLayerUI(){
    const all=byId("traceTargetAll"),track=byId("traceTargetTrack"),add=byId("traceTargetNew");if(!all||!track||!add)return;
    const t=trackById(layerState.trackId)||trackById(layerState.lastTrackId);
    all.classList.toggle("active",layerState.mode==="all");track.classList.toggle("active",layerState.mode==="track");add.classList.toggle("active",layerState.mode==="layer");
    track.textContent=t?`CH ${t.id} · ${t.name}`:"TRACK";
    const scope=scopeObject().label.replace("WHOLE SONG","ALL");
    byId("scopeLabel").textContent=layerState.mode==="track"&&t?`${scope} · CH ${t.id}`:layerState.mode==="layer"?`${scope} · NEW PART`:scopeObject().label;
    const input=byId("prompt");if(input)input.placeholder=layerState.mode==="track"&&t?`Change only ${t.name}…`:layerState.mode==="layer"?"Describe the next part to layer in…":"Make, ask why, ask otherwise, or describe what you hear…";
  }

  globalThis.renderTimeline=function(){
    priorRenderTimeline();
    const tracks=v8ActiveTracks(current()),labels=[...document.querySelectorAll("#timeline .v8VoiceLabel")];
    labels.forEach((lab,i)=>{const t=tracks[i];if(!t)return;lab.classList.toggle("traceFocused",layerState.mode==="track"&&+layerState.trackId===+t.id);lab.title=layerState.mode==="track"&&+layerState.trackId===+t.id?"Focused for prompting · tap again to choose voice":"Focus this track for prompting";lab.onclick=e=>{e.preventDefault();if(layerState.mode==="track"&&+layerState.trackId===+t.id)v8OpenVoice(+t.id);else setLayerMode("track",+t.id)}});
  };
  globalThis.renderMessages=function(){
    priorRenderMessages();const root=byId("messages");if(!root)return;
    [...root.children].forEach((el,i)=>{const m=state.messages[i],target=m?.plan?.trace_target;if(!target||target.mode==="all")return;const d=document.createElement("div");d.className=`layerPlanTarget ${target.mode==="layer"?"add":""}`;d.textContent=target.mode==="layer"?`ADD · CH ${target.track} · ${target.scope?.label||"SCOPE"}`:`ONLY · CH ${target.track} · ${target.name||"TRACK"} · ${target.scope?.label||"SCOPE"}`;const who=el.querySelector(".who");who?.insertAdjacentElement("afterend",d)});
  };
  globalThis.renderAll=function(){priorRenderAll();installLayerUI();syncLayerUI()};
  globalThis.applyMessage=function(id){
    const m=state.messages.find(x=>x.id===id&&x.plan),plan=m?.variants?.[m.variantIndex||0]||m?.plan,target=plan?.trace_target;priorApplyMessage(id);
    if(!target||target.mode==="all")return;
    const r=state.runs.find(x=>+x.id===+state.currentRunId);if(r){r.meta={...(r.meta||{}),target:clone(target)};r.label=target.mode==="layer"?`layer CH ${target.track}`:`edit CH ${target.track}`}
    layerState.lastTrackId=+target.track;layerState.mode="track";layerState.trackId=+target.track;saveSession();renderAll();renderTrail();
    showCoach(target.mode==="layer"?"PART ADDED":"TRACK WRITTEN",target.mode==="layer"?`CH ${target.track} joined the song. Prompt it directly now, or tap + PART for another layer.`:`Only CH ${target.track} was rewritten inside the captured time scope.`);
  };

  globalThis.v8Init=function(){priorInit();installLayerUI();syncLayerUI();renderAll()};
})();
