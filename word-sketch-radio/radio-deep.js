'use strict';

// SKETCHRADIO DEEP PASS — raw-audio listening + visible relation patches.
const RADIO_AUDIO_MODEL='gpt-realtime-2.1-mini';
const WYGWYL_AUDIO='https://hartswf0.github.io/prompt-language/WYGWYL-BK/OP-51/wygwyl-site/audio/unified-drones.mp3';
let radioHearing={source:'none',raw:'',transcript:'',performance:'',time:'',score:'',windowStart:0,windowEnd:0};
let radioRelations=[];
let radioUndo=[];

function installDeepRadioStyles(){
  if($('#radioDeepStyles'))return;
  const s=document.createElement('style');
  s.id='radioDeepStyles';
  s.textContent=`
  :root{--transport:96px}
  .transport{grid-template-columns:58px minmax(0,1fr) 104px;grid-template-rows:38px 58px}
  .radioCommand{grid-column:1/-1;grid-row:1;display:grid;grid-template-columns:112px minmax(0,1fr) 74px;border-bottom:1px solid var(--ink);min-width:0;background:var(--paper)}
  .radioCommand select,.radioCommand input,.radioCommand button{border:0;background:var(--paper);min-width:0;font:800 8px/1 var(--mono)}
  .radioCommand select{border-right:1px solid var(--ink);padding:0 7px}
  .radioCommand input{padding:0 9px;outline:0}
  .radioCommand button{border-left:1px solid var(--ink);background:var(--signal);color:#fff;font-weight:900}
  .playBtn,.scrubBox,.makeBtn{grid-row:2}
  .playBtn{grid-column:1}.scrubBox{grid-column:2}.makeBtn{grid-column:3}
  .radioSource{height:34px;flex:0 0 34px;display:grid;grid-template-columns:auto auto auto auto minmax(50px,1fr);border-bottom:1px solid rgba(8,10,8,.35);min-width:0}
  .radioSource button{border:0;border-right:1px solid rgba(8,10,8,.35);background:var(--paper);padding:0 7px;font:900 7px/1 var(--mono);white-space:nowrap}
  .radioSource button:disabled{color:#aaa;background:#f3f1e8}.radioSource button:not(:disabled):active{background:var(--ink);color:var(--paper)}
  #radioWygwyl{background:var(--acid)}
  #radioSourceState{display:flex;align-items:center;padding:0 7px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:800 7px/1 var(--mono);color:var(--muted)}
  .radioReading{height:48px;flex:0 0 48px;padding:6px 8px;border-bottom:1px solid rgba(8,10,8,.25);overflow:hidden;font:700 8px/1.35 var(--mono);color:#393a34;background:#fff}
  #radioTimeInk,#radioWorldCursor{pointer-events:none;z-index:2}
  #radioTimeInk{opacity:1}
  .radioRelationBand{height:76px;flex:0 0 76px;border-bottom:1px solid rgba(8,10,8,.35);display:grid;grid-template-rows:24px minmax(0,1fr);background:#fff;min-width:0}
  .radioRelationHead{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:7px;padding:0 6px;border-bottom:1px solid rgba(8,10,8,.22);font:900 7px/1 var(--mono)}
  .radioRelationHead span:nth-child(2){color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .radioRelationHead button{height:19px;border:1px solid rgba(8,10,8,.35);background:var(--paper);font:900 6px/1 var(--mono)}
  .radioRelationList{display:flex;min-width:0;overflow-x:auto;overflow-y:hidden;scrollbar-width:none}
  .radioRelationList::-webkit-scrollbar{display:none}
  .radioRelation{flex:0 0 min(210px,58vw);display:grid;grid-template-columns:minmax(0,1fr) 18px minmax(0,1fr);align-items:center;border:0;border-right:1px solid rgba(8,10,8,.25);background:#fff;padding:4px 6px;text-align:left;min-width:0}
  .radioRelation b{font:900 9px/1 var(--sans);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.radioRelation em{font:900 10px/1 var(--mono);font-style:normal;text-align:center}.radioRelation small{display:block;margin-top:3px;font:700 6px/1.15 var(--mono);color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .radioRelation:active{background:var(--acid)}
  .radioNoRelations{display:grid;place-items:center;min-width:100%;font:900 7px/1 var(--mono);letter-spacing:.08em;color:#aaa}
  .radioHearTag{font-weight:900;color:var(--good)}
  @media(max-width:900px){
    :root{--transport:94px}
    .radioSource{grid-template-columns:repeat(4,auto) minmax(40px,1fr);height:32px;flex-basis:32px}
    .radioSource button{font-size:6px;padding:0 5px}
    .radioReading{height:42px;flex-basis:42px;font-size:7px;padding:5px 7px}
    .radioRelationBand{height:70px;flex-basis:70px}
    .radioCommand{grid-template-columns:92px minmax(0,1fr) 62px}
    .radioCommand select,.radioCommand input,.radioCommand button{font-size:7px}
    .radioRelation{flex-basis:68vw}
  }`;
  document.head.appendChild(s);
}

function installDeepRadioUI(){
  installDeepRadioStyles();
  const source=$('.radioSource');
  if(source&&!$('#radioWygwyl')){
    const b=document.createElement('button');
    b.id='radioWygwyl';b.textContent='WYGWYL';b.title='Load the WYGWYL poems + drone as a source';
    source.insertBefore(b,$('#radioLoad'));
    b.onclick=loadWygwylSource;
  }
  const panel=$('#codePanel');
  if(panel&&!$('#radioRelationBand')){
    const band=document.createElement('div');
    band.className='radioRelationBand';band.id='radioRelationBand';
    band.innerHTML=`<div class="radioRelationHead"><span>RELATIONS</span><span id="radioRelationStatus">HEAR OR SHAPE TO MAP</span><button id="radioUndo" disabled>UNDO</button></div><div class="radioRelationList" id="radioRelationList"></div>`;
    panel.querySelector('.ph').after(band);
    $('#radioUndo').onclick=undoRadioPatch;
  }
  renderRadioRelations();
}

async function loadWygwylSource(){
  busy(true,'LOADING WYGWYL');
  try{
    const r=await fetch(WYGWYL_AUDIO,{cache:'force-cache'});
    if(!r.ok)throw new Error(`WYGWYL ${r.status}`);
    const blob=await r.blob();
    let f;
    try{f=new File([blob],'WYGWYL poems + drone.mp3',{type:blob.type||'audio/mpeg'})}
    catch{blob.name='WYGWYL poems + drone.mp3';f=blob}
    await loadVoiceFile(f);
    $('#radioSourceState').textContent=`WYGWYL · ${fmtTime(radioVoice.duration)}`;
    setState('WYGWYL READY · PLAY / SCRUB / HEAR','good');
  }catch(e){setState(e.message||'WYGWYL LOAD FAILED','bad')}
  finally{busy(false)}
}

function bytesToBase64(bytes){
  let out='',chunk=0x8000;
  for(let i=0;i<bytes.length;i+=chunk)out+=String.fromCharCode(...bytes.subarray(i,Math.min(bytes.length,i+chunk)));
  return btoa(out);
}
function wavWindowBase64(buffer,centerSec=0,maxSeconds=30,targetRate=16000){
  const dur=Math.min(maxSeconds,buffer.duration);
  let start=Math.max(0,centerSec-6);
  if(start+dur>buffer.duration)start=Math.max(0,buffer.duration-dur);
  const frames=Math.max(1,Math.floor(dur*targetRate));
  const pcm=new Int16Array(frames);
  const ratio=buffer.sampleRate/targetRate;
  const startFrame=start*buffer.sampleRate;
  for(let i=0;i<frames;i++){
    const src=Math.min(buffer.length-1,Math.floor(startFrame+i*ratio));
    let v=0;
    for(let ch=0;ch<buffer.numberOfChannels;ch++)v+=buffer.getChannelData(ch)[src]||0;
    v=Math.max(-1,Math.min(1,v/buffer.numberOfChannels));
    pcm[i]=v<0?v*32768:v*32767;
  }
  const ab=new ArrayBuffer(44+pcm.length*2),dv=new DataView(ab);
  const write=(o,s)=>{for(let i=0;i<s.length;i++)dv.setUint8(o+i,s.charCodeAt(i))};
  write(0,'RIFF');dv.setUint32(4,36+pcm.length*2,true);write(8,'WAVE');write(12,'fmt ');dv.setUint32(16,16,true);dv.setUint16(20,1,true);dv.setUint16(22,1,true);dv.setUint32(24,targetRate,true);dv.setUint32(28,targetRate*2,true);dv.setUint16(32,2,true);dv.setUint16(34,16,true);write(36,'data');dv.setUint32(40,pcm.length*2,true);
  for(let i=0;i<pcm.length;i++)dv.setInt16(44+i*2,pcm[i],true);
  return{data:bytesToBase64(new Uint8Array(ab)),format:'wav',start,end:start+dur};
}
function hearingField(text,label){
  const re=new RegExp(`${label}:\\s*([\\s\\S]*?)(?=\\n(?:TRANSCRIPT|PERFORMANCE|TIME|SCORE):|$)`,'i');
  return (re.exec(text)?.[1]||'').trim();
}

const shallowDescribeCurrentSound=describeCurrentSound;
describeCurrentSound=async function(){
  if(!radioVoice.buffer)return;
  if(!verified){$('#gate').classList.remove('hidden');return}
  busy(true,'HEARING RAW AUDIO');
  try{
    const center=radioVoice.audio?.currentTime||radioTimeFromBeat();
    const clip=wavWindowBase64(radioVoice.buffer,center,30,16000);
    const request=`Listen to this actual voice/radio audio as a film scorer and temporal editor.
Return exactly four labeled sections and nothing else:
TRANSCRIPT: words you can confidently hear; mark uncertainty briefly rather than inventing.
PERFORMANCE: breath, cadence, pressure, repetition, vocal grain, acceleration/deceleration, and expressive contour.
TIME: where silence, phrase endings, attacks, held sounds, or recurring pulses create usable entrances. Use relative descriptions; exact timestamps only when evident.
SCORE: one restrained accompaniment move that could live beside this voice without forcing it onto a grid.
The source may contain voice plus music/drone. Distinguish them when possible.`;
    const data=await apiFetch('/responses',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      model:RADIO_AUDIO_MODEL,
      input:[{role:'user',content:[{type:'input_text',text:request},{type:'input_audio',input_audio:{data:clip.data,format:clip.format}}]}],
      max_output_tokens:700
    })});
    const raw=responseText(data).trim();
    radioHearing={source:'raw-audio',raw,transcript:hearingField(raw,'TRANSCRIPT'),performance:hearingField(raw,'PERFORMANCE'),time:hearingField(raw,'TIME'),score:hearingField(raw,'SCORE'),windowStart:clip.start,windowEnd:clip.end};
    radioVoice.reading=`RAW AUDIO ${fmtTime(clip.start)}–${fmtTime(clip.end)} · ${radioHearing.performance||raw} · ${radioHearing.time||''} · ${radioHearing.score||''}`;
    if(!$('#wordInput').value.trim()&&radioHearing.transcript){$('#wordInput').value=radioHearing.transcript.slice(0,900);fitWord();markDirty('word',true)}
    $('#radioReading').innerHTML=`<span class="radioHearTag">RAW AUDIO HEARD ${fmtTime(clip.start)}–${fmtTime(clip.end)}</span> · ${escapeHtml(radioHearing.performance||radioHearing.transcript||raw)}`;
    radioRenderWorld();
    await compileRadioRelations('Map three strong correspondences from what you just heard. Keep the voice sovereign and leave silence available.',false);
    setState('RAW AUDIO HEARD · RELATIONS MAPPED','good');
  }catch(e){
    radioHearing={source:'sound-map-fallback',raw:'',transcript:'',performance:'',time:'',score:'',windowStart:0,windowEnd:0};
    setState('RAW AUDIO MODEL UNAVAILABLE · USING SOUND MAP','bad');
    await shallowDescribeCurrentSound();
    try{await compileRadioRelations('Map three restrained correspondences from the sound map. Do not invent words or pitch.',false)}catch{}
  }finally{busy(false)}
};

const RELATION_PATCH_SCHEMA={type:'object',additionalProperties:false,required:['summary','relations'],properties:{
  summary:{type:'string'},
  relations:{type:'array',minItems:2,maxItems:5,items:{type:'object',additionalProperties:false,required:['from','to','operation','strength','lag_seconds','reason'],properties:{
    from:{type:'string'},to:{type:'string'},operation:{type:'string'},strength:{type:'number',minimum:0,maximum:1},lag_seconds:{type:'number',minimum:-8,maximum:8},reason:{type:'string'}
  }}}
}};
function relationText(r){return `${r.from} → ${r.to}: ${r.operation}; strength ${Number(r.strength||0).toFixed(2)}; lag ${Number(r.lag_seconds||0).toFixed(2)}s`}
function relationGraphText(){return radioRelations.length?radioRelations.map(relationText).join('\n'):'(none yet)'}
async function compileRadioRelations(request,pushUndo=true){
  if(!verified)return;
  if(pushUndo)pushRadioUndo();
  radioRenderWorld();
  const prompt=`You are the relation compiler inside SKETCHRADIO. Convert a human scoring instruction into 2-5 inspectable causal correspondences.
A relation may connect POEM, VOICE, BREATH, SILENCE, PULSE, TIMED_MARK, WORLD, SKY, LAND, SCORE, HARMONY, RHYTHM, TIMBRE, CAMERA, or RADIO.
Preserve source identities. The recorded voice is sovereign unless the request explicitly asks to edit it.
Prefer restrained relationships, delayed answers, selective alignment, and silence over constant synchronization.
A negative lag means anticipation. A positive lag means response after the source.

USER REQUEST: ${request||'(find useful relations)'}
SCORING MODE: ${radioMode} — ${RADIO_MODE_INSTRUCTIONS[radioMode]}
POEM: ${$('#wordInput').value.trim()||'(none)'}
RAW HEARING: ${radioHearing.raw||radioVoice.reading||voiceSummary()}
TIMED DRAWING: ${timedSketchSummary()}
CURRENT RELATIONS:\n${relationGraphText()}`;
  const images=[];
  if(baseHasSketchInk())images.push({label:'TIMED SKETCH STATE',url:canvasDataURL(sketch)});
  if(radioVoice.buffer)images.push({label:'VOICE / WORLD MAP',url:canvasDataURL(specBase)});
  const out=await structured(contentForImages(prompt,images),'sketchradio_relation_patch',RELATION_PATCH_SCHEMA);
  radioRelations=out.relations;
  renderRadioRelations(out.summary);
  return out;
}
function renderRadioRelations(summary=''){
  const box=$('#radioRelationList');if(!box)return;
  $('#radioRelationStatus').textContent=summary||`${radioRelations.length||0} ACTIVE · TAP ONE TO REVERSE`;
  $('#radioUndo').disabled=!radioUndo.length;
  if(!radioRelations.length){box.innerHTML='<div class="radioNoRelations">VOICE · MARK · WORLD · SCORE</div>';return}
  box.innerHTML=radioRelations.map((r,i)=>`<button class="radioRelation" data-rel="${i}" title="Tap to reverse this causal arrow"><span><b>${escapeHtml(r.from)}</b><small>${escapeHtml(r.reason)}</small></span><em>→</em><span><b>${escapeHtml(r.to)}</b><small>${escapeHtml(r.operation)} · ${Math.round(r.strength*100)}% · ${r.lag_seconds>=0?'+':''}${r.lag_seconds.toFixed(1)}s</small></span></button>`).join('');
  $$('[data-rel]').forEach(b=>b.onclick=()=>reverseRadioRelation(Number(b.dataset.rel)));
}
function pushRadioUndo(){
  radioUndo.push({relations:structuredCloneSafe(radioRelations),pipeline:structuredCloneSafe(pipeline),song:structuredCloneSafe(song),mode:radioMode});
  if(radioUndo.length>10)radioUndo.shift();
  $('#radioUndo')?.removeAttribute('disabled');
}
async function undoRadioPatch(){
  const s=radioUndo.pop();if(!s)return;
  stop();radioRelations=s.relations||[];pipeline=s.pipeline;song=s.song;radioMode=s.mode||radioMode;
  if($('#radioMode'))$('#radioMode').value=radioMode;
  if(song)await renderSong();else{sv.clearRect(0,0,songViz.width,songViz.height);$('#songTitle').textContent='NO SCORE';}
  radioRenderWorld();renderRadioRelations('RESTORED PREVIOUS RELATION STATE');updateMakeState();setState('UNDO · PREVIOUS WORLD RESTORED','good');
}
function reverseRadioRelation(i){
  const r=radioRelations[i];if(!r)return;
  pushRadioUndo();
  [r.from,r.to]=[r.to,r.from];
  r.operation=`reverse: ${r.operation}`;
  renderRadioRelations('ARROW REVERSED · SHAPE TO AUDITION');
  $('#radioPrompt').value=`Audition the reversed relation ${r.from} → ${r.to}. Preserve the other relations.`;
  $('#changeInput').value=$('#radioPrompt').value;markDirty('change',true);setState('RELATION REVERSED · SHAPE TO HEAR','good');
}

const shallowRadioTheoryPrompt=radioTheoryPrompt;
radioTheoryPrompt=function(revise=false){
  return `${shallowRadioTheoryPrompt(revise)}\n\nVISIBLE RELATION GRAPH — THESE ARE THE CURRENT PATCH CABLES:\n${relationGraphText()}\n\nRAW-AUDIO HEARING:\n${radioHearing.raw||'(not heard directly yet)'}\n\nMake the score AUDITION these relations. Do not silently replace them with generic style.`;
};

const shallowRadioMakeOrLoop=radioMakeOrLoop;
async function deepRadioShape(){
  if(!verified){$('#gate').classList.remove('hidden');return}
  const req=$('#radioPrompt')?.value.trim()||$('#changeInput').value.trim();
  try{
    if(req)await compileRadioRelations(req,true);
    else if(!radioRelations.length)await compileRadioRelations(`Build a restrained ${radioMode.toLowerCase()} relation between the current sources.`,true);
    await shallowRadioMakeOrLoop();
    renderRadioRelations('AUDITIONING CURRENT RELATIONS');
  }catch(e){setState(e.message||'SHAPE FAILED','bad')}
}

function patchDeepBindings(){
  $('#radioHear').onclick=describeCurrentSound;
  $('#radioShape').onclick=deepRadioShape;
  $('#makeBtn').onclick=deepRadioShape;
  $('#radioPrompt').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();deepRadioShape()}};
  const oldLoad=$('#radioFile').onchange;
  $('#radioFile').onchange=async e=>{
    radioHearing={source:'none',raw:'',transcript:'',performance:'',time:'',score:'',windowStart:0,windowEnd:0};radioRelations=[];radioUndo=[];renderRadioRelations();
    await oldLoad?.call($('#radioFile'),e);
  };
}

// Add raw hearing + relation graph to the existing guide context.
const deepBaseGuide=guideContextPrompt;
guideContextPrompt=function(mode,q){return `${deepBaseGuide(mode,q)}\nRAW AUDIO HEARING: ${radioHearing.raw||'(none)'}\nVISIBLE RELATIONS:\n${relationGraphText()}\nIf the user asks to shape the scene, answer in terms of which causal arrow should change.`};

installDeepRadioUI();
patchDeepBindings();
