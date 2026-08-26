'use strict';

let radioVoice={file:null,url:'',buffer:null,audio:null,duration:0,analysis:null,reading:''};
let radioMode='HARMONIZE';
let timedStrokes=[],activeTimedStroke=null,timedBaseImage=null,timedBaseURL='';
let radioVoiceRaf=0;

function radioEl(tag,cls='',text=''){
  const e=document.createElement(tag);
  if(cls)e.className=cls;
  if(text)e.textContent=text;
  return e;
}
function fmtTime(s){
  s=Math.max(0,Number(s)||0);
  return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
}
function radioTimeFromBeat(b=currentBeat){
  if(radioVoice.duration)return Math.max(0,Math.min(radioVoice.duration,(b/48)*radioVoice.duration));
  if(song)return Math.max(0,b*60/song.tempo_bpm);
  return 0;
}
function radioClockTime(){
  if(radioVoice.audio)return radioVoice.audio.currentTime||radioTimeFromBeat();
  return radioTimeFromBeat();
}
function voiceSummary(){
  const a=radioVoice.analysis;
  if(!a)return '(no voice/radio sound loaded)';
  const contour=a.envelope.slice(0,12).map(v=>Math.round(v*9)).join('');
  return `duration ${a.duration.toFixed(1)}s; average energy ${a.rms.toFixed(3)}; silence ${(a.silenceRatio*100).toFixed(0)}%; peak ${a.peak.toFixed(2)}; envelope ${contour}`;
}
function timedSketchSummary(){
  if(!timedStrokes.length)return '(no timed strokes)';
  const pts=timedStrokes.flatMap(s=>s.points);
  const t0=Math.min(...pts.map(p=>p.t)),t1=Math.max(...pts.map(p=>p.t));
  let rise=0,fall=0,travel=0;
  for(const s of timedStrokes)for(let i=1;i<s.points.length;i++){
    const a=s.points[i-1],b=s.points[i],dy=b.y-a.y,dx=b.x-a.x;
    if(dy<0)rise+=Math.abs(dy);else fall+=dy;
    travel+=Math.hypot(dx,dy);
  }
  return `${timedStrokes.length} performed strokes from ${t0.toFixed(1)}s to ${t1.toFixed(1)}s; travel ${Math.round(travel)}px; upward ${Math.round(rise)} / downward ${Math.round(fall)}`;
}
const RADIO_MODE_INSTRUCTIONS={
  HARMONIZE:'Find harmonic space around the voice. Support its cadence without simply doubling pitch. Preserve room for words.',
  ANSWER:'Treat phrase endings and pauses as entrances. Let the score answer rather than talk over the voice.',
  SHADOW:'Track contour, density and stress quietly, with delayed or partial doubling rather than literal imitation.',
  COUNTERPOINT:'Build an independent musical line whose occasional coincidences with the voice feel earned.',
  RESIST:'Keep a stable pulse or harmonic pressure against free speech. Do not quantize the voice.',
  EMBODY:'Translate voice and drawing into world behavior and timbre before adding notes. Music can emerge from the environment.'
};

function installRadioSurface(){
  document.title='Sketchradio — Word · Voice · Sketch · World';
  $('.brand span:last-child').textContent='SKETCHRADIO';
  const labels=$$('.columnLabel');
  if(labels[0])labels[0].innerHTML='<b>FROM</b><span>poem + voice + mark</span>';
  if(labels[1])labels[1].innerHTML='<b>THROUGH</b><span>prompt the relation</span>';
  if(labels[2])labels[2].innerHTML='<b>TO</b><span>world + soundtrack</span>';

  const wordHead=$('#wordPanel .ph span:nth-of-type(1)');
  if(wordHead)wordHead.textContent='POEM';
  const sketchHead=$('#sketchPanel .ph span:nth-of-type(1)');
  if(sketchHead)sketchHead.textContent='TIMED SKETCH';
  const songHead=$('#songPanel .ph span:nth-of-type(1)');
  if(songHead)songHead.textContent='SCORE';
  const specHead=$('#spectrumPanel .ph span:nth-of-type(1)');
  if(specHead)specHead.textContent='WORLD / SOUND';
  $('#wordInput').placeholder='poem, line, instruction, fragment';

  const source=radioEl('div','radioSource');
  source.innerHTML=`<input id="radioFile" type="file" accept="audio/*" class="hidden">
    <button id="radioLoad">LOAD VOICE / RADIO</button>
    <button id="radioHear" disabled>HEAR</button>
    <button id="radioSketchSound" disabled>SOUND → SKETCH</button>
    <span id="radioSourceState">NO SOUND</span>`;
  $('#wordPanel .ph').after(source);
  const reading=radioEl('div','radioReading','AI LISTENING APPEARS HERE');
  reading.id='radioReading';
  source.after(reading);

  const timeInk=document.createElement('canvas');
  timeInk.id='radioTimeInk';timeInk.width=1024;timeInk.height=1024;
  $('#sketchWrap').appendChild(timeInk);

  const cursor=document.createElement('canvas');
  cursor.id='radioWorldCursor';cursor.width=1024;cursor.height=512;
  $('#specWrap').appendChild(cursor);

  const command=radioEl('div','radioCommand');
  command.innerHTML=`<select id="radioMode" aria-label="scoring relation">
      ${Object.keys(RADIO_MODE_INSTRUCTIONS).map(x=>`<option>${x}</option>`).join('')}
    </select>
    <input id="radioPrompt" maxlength="500" placeholder="ask the soundtrack / world…">
    <button id="radioShape">SHAPE</button>`;
  $('.transport').prepend(command);

  $('#radioLoad').onclick=()=>$('#radioFile').click();
  $('#radioFile').onchange=e=>loadVoiceFile(e.target.files?.[0]);
  $('#radioHear').onclick=describeCurrentSound;
  $('#radioSketchSound').onclick=soundToSketch;
  $('#radioMode').onchange=e=>{radioMode=e.target.value;setState(`${radioMode} · READY TO SHAPE`,'good')};
  $('#radioPrompt').addEventListener('input',e=>{
    $('#changeInput').value=e.target.value;
    markDirty('change',!!e.target.value.trim());
  });
  $('#radioPrompt').addEventListener('keydown',e=>{
    if(e.key==='Enter'){e.preventDefault();radioMakeOrLoop()}
  });
  $('#radioShape').onclick=radioMakeOrLoop;

  captureTimedDrawing();
  wireRadioTransport();
  installRadioAIPatches();
  updateMakeState();
}

async function loadVoiceFile(file){
  if(!file)return;
  stop();
  if(radioVoice.url)URL.revokeObjectURL(radioVoice.url);
  const ctx=ensureAudio();
  busy(true,'LISTENING');
  try{
    const arr=await file.arrayBuffer();
    const buffer=await ctx.decodeAudioData(arr.slice(0));
    const url=URL.createObjectURL(file),audio=new Audio(url);
    audio.preload='auto';
    radioVoice={file,url,buffer,audio,duration:buffer.duration,analysis:analyzeVoice(buffer),reading:''};
    currentBeat=0;
    $('#radioSourceState').textContent=`${file.name} · ${fmtTime(buffer.duration)}`;
    $('#radioHear').disabled=false;$('#radioSketchSound').disabled=false;
    $('#radioReading').textContent='SOUND LOADED · HEAR asks AI to read it · draw while it plays';
    radioRenderWorld();
    updateTransport();
    updateMakeState();
    if(typeof traceNode==='function'){
      traceNode('spectrum','VOICE / RADIO',voiceSummary());
      persistTrace?.();refreshFlow?.();
    }
    setState('VOICE / RADIO READY','good');
  }catch(e){setState(e.message||'AUDIO LOAD FAILED','bad')}
  finally{busy(false)}
}
function analyzeVoice(buffer){
  const channels=[];
  for(let c=0;c<buffer.numberOfChannels;c++)channels.push(buffer.getChannelData(c));
  const n=buffer.length,step=Math.max(1,Math.floor(n/24000));
  let sum=0,peak=0,count=0;
  for(let i=0;i<n;i+=step){
    let v=0;for(const ch of channels)v+=ch[i]||0;v/=channels.length;
    sum+=v*v;peak=Math.max(peak,Math.abs(v));count++;
  }
  const rms=Math.sqrt(sum/Math.max(1,count)),bins=128,envelope=[],silences=[];
  for(let b=0;b<bins;b++){
    const a=Math.floor(b*n/bins),z=Math.floor((b+1)*n/bins),s=Math.max(1,Math.floor((z-a)/500));
    let ss=0,cc=0;
    for(let i=a;i<z;i+=s){
      let v=0;for(const ch of channels)v+=ch[i]||0;v/=channels.length;
      ss+=v*v;cc++;
    }
    const r=Math.sqrt(ss/Math.max(1,cc));envelope.push(r);
  }
  const max=Math.max(...envelope,1e-6),norm=envelope.map(v=>v/max);
  const threshold=Math.max(.035,rms*.32);
  for(const v of envelope)silences.push(v<threshold);
  return{duration:buffer.duration,rms,peak,envelope:norm,silences,silenceRatio:silences.filter(Boolean).length/bins};
}
async function describeCurrentSound(){
  if(!radioVoice.buffer)return;
  if(!verified){$('#gate').classList.remove('hidden');return}
  busy(true,'AI LISTENING');
  try{
    radioRenderWorld();
    const schema={type:'object',additionalProperties:false,required:['reading','cadence','space','score_move'],properties:{
      reading:{type:'string'},cadence:{type:'string'},space:{type:'string'},score_move:{type:'string'}
    }};
    const prompt=`Read this visual sound-map as evidence of a real voice/radio recording, not as decoration.
Describe what a composer or drawer can OPERATE ON: cadence, density, silence, contour, recurrence, pressure and room.
Do not invent exact pitch or words you cannot know. Return a compact sonic reading and one useful scoring move.
LOCAL ANALYSIS: ${voiceSummary()}`;
    const c=contentForImages(prompt,[{label:'CURRENT SOUND MAP',url:canvasDataURL(specBase)}]);
    const r=await structured(c,'sketchradio_sound_reading',schema);
    radioVoice.reading=`${r.reading} Cadence: ${r.cadence}. Space: ${r.space}. Score move: ${r.score_move}`;
    $('#radioReading').textContent=radioVoice.reading;
    setState('AI HEARD THE SOUND MAP','good');
  }catch(e){setState(e.message,'bad')}
  finally{busy(false)}
}
async function soundToSketch(){
  if(!radioVoice.buffer)return;
  if(!verified){$('#gate').classList.remove('hidden');return}
  busy(true,'SOUND → SKETCH');
  try{
    if(!radioVoice.reading)await describeCurrentSound();
    const content=contentForImages(`Turn this sound-map into an image prompt for a rough black graphite / ink thinking-sketch on bright paper.
The drawing must be useful as a control surface: time runs left to right; height may imply register; density may imply activity; empty regions must preserve silence.
No readable text, no staff notation, no decorative illustration.
SOUND READING: ${radioVoice.reading||voiceSummary()}`,[{label:'SOUND MAP',url:canvasDataURL(specBase)}]);
    const rr=await apiFetch('/responses',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:$('#textModel').value,input:[{role:'user',content}]})});
    const prompt=responseText(rr).trim();
    const data=await apiFetch('/images/generations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:$('#imageModel').value,prompt,size:'1024x1024',quality:'medium'})});
    const item=data.data?.[0];if(!item)throw new Error('No image returned');
    const src=item.b64_json?`data:image/png;base64,${item.b64_json}`:item.url;
    await loadSketchImage(src);
    timedStrokes=[];timedBaseImage=null;timedBaseURL='';$('#sketch').style.opacity='1';
    const ti=$('#radioTimeInk').getContext('2d');ti.clearRect(0,0,1024,1024);
    markDirty('sketch',true);
    traceTransform?.('spectrum','sketch','SOUND→SKETCH');
    setState('SOUND BECAME A SKETCH','good');
  }catch(e){setState(e.message,'bad')}
  finally{busy(false)}
}

function captureTimedDrawing(){
  const c=sketch;
  c.addEventListener('pointerdown',e=>{
    if(!timedStrokes.length)captureTimedBase();
    const p=point(c,e),t=radioClockTime();
    activeTimedStroke={t0:t,t1:t,points:[{x:p.x,y:p.y,t,pressure:e.pressure||.5}]};
    timedStrokes.push(activeTimedStroke);
  },true);
  c.addEventListener('pointermove',e=>{
    if(!activeTimedStroke)return;
    const p=point(c,e),t=radioClockTime();
    activeTimedStroke.points.push({x:p.x,y:p.y,t,pressure:e.pressure||.5});
    activeTimedStroke.t1=t;
    requestAnimationFrame(()=>redrawTimedSketch(t));
  },true);
  const end=()=>{
    if(!activeTimedStroke)return;
    activeTimedStroke.t1=radioClockTime();
    activeTimedStroke=null;
    redrawTimedSketch(radioClockTime());
  };
  c.addEventListener('pointerup',end,true);c.addEventListener('pointercancel',end,true);
  $('#clearSketch').addEventListener('click',()=>{
    timedStrokes=[];activeTimedStroke=null;timedBaseImage=null;timedBaseURL='';
    $('#radioTimeInk').getContext('2d').clearRect(0,0,1024,1024);
    sketch.style.opacity='1';
  });
}
function captureTimedBase(){
  try{
    timedBaseURL=sketch.toDataURL('image/png');
    const im=new Image();
    im.onload=()=>{timedBaseImage=im;redrawTimedSketch(radioClockTime())};
    im.src=timedBaseURL;
  }catch{}
}
function redrawTimedSketch(t){
  if(!timedStrokes.length)return;
  const c=$('#radioTimeInk'),x=c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);
  if(timedBaseImage)x.drawImage(timedBaseImage,0,0,c.width,c.height);
  x.strokeStyle='#0b0c0b';x.lineWidth=7;x.lineCap='round';x.lineJoin='round';
  for(const s of timedStrokes){
    const pts=s.points;
    if(!pts.length||pts[0].t>t)continue;
    x.beginPath();x.moveTo(pts[0].x,pts[0].y);
    for(let i=1;i<pts.length;i++){
      if(pts[i].t>t)break;
      x.lineTo(pts[i].x,pts[i].y);
    }
    x.stroke();
  }
  sketch.style.opacity='0';
}
function radioRenderWorld(){
  const w=1024,h=512;
  sb.clearRect(0,0,w,h);sb.fillStyle='#fffdf4';sb.fillRect(0,0,w,h);
  sb.strokeStyle='rgba(8,10,8,.08)';sb.lineWidth=1;
  for(let i=0;i<=12;i++){const x=i/12*w;sb.beginPath();sb.moveTo(x,0);sb.lineTo(x,h);sb.stroke()}
  if(radioVoice.analysis){
    const a=radioVoice.analysis,N=a.envelope.length;
    for(let i=0;i<N;i++)if(a.silences[i]){
      sb.fillStyle='rgba(236,255,99,.34)';sb.fillRect(i/N*w,0,w/N+1,h);
    }
    sb.beginPath();sb.moveTo(0,h*.78);
    for(let i=0;i<N;i++){
      const x=i/(N-1)*w,y=h*.78-a.envelope[i]*h*.46;
      sb.lineTo(x,y);
    }
    sb.lineTo(w,h);sb.lineTo(0,h);sb.closePath();
    sb.fillStyle='rgba(8,10,8,.12)';sb.fill();
    sb.strokeStyle='#080a08';sb.lineWidth=2;sb.stroke();
  }
  if(song){
    for(const ev of flattenEvents()){
      const x=ev.offset/48*w,ww=Math.max(2,ev.duration/48*w),m=pitchMidi(ev.pitch);
      const y=h*.08+(96-m)/(96-36)*h*.38;
      sb.fillStyle=ev.section===1?'#ff522d':'#080a08';
      sb.fillRect(x,y,ww,4);
    }
  }
  $('#specEmpty').classList.add('hidden');
  drawRadioCursor();
}
function drawRadioCursor(){
  const c=$('#radioWorldCursor');if(!c)return;
  const x=c.getContext('2d'),w=c.width,h=c.height;
  x.clearRect(0,0,w,h);
  const px=Math.max(0,Math.min(w,currentBeat/48*w));
  x.fillStyle='#0b7d52';x.fillRect(px,0,2,h);
}

function wireRadioTransport(){
  const basePlay=play,basePause=pause,baseStop=stop,baseUpdate=updateTransport;
  play=function(){
    if(song){
      if(radioVoice.audio){
        radioVoice.audio.currentTime=radioTimeFromBeat();
        radioVoice.audio.play().catch(()=>{});
      }
      basePlay();
      return;
    }
    if(!radioVoice.audio)return;
    playing=true;
    radioVoice.audio.currentTime=radioTimeFromBeat();
    radioVoice.audio.play().catch(()=>{});
    iconUse($('#playUse'),'#i-pause');
    cancelAnimationFrame(radioVoiceRaf);
    const tickVoice=()=>{
      if(!playing||song)return;
      currentBeat=Math.min(48,(radioVoice.audio.currentTime/radioVoice.duration)*48);
      updateTransport();
      if(radioVoice.audio.ended){playing=false;currentBeat=48;iconUse($('#playUse'),'#i-play');updateTransport();return}
      radioVoiceRaf=requestAnimationFrame(tickVoice);
    };
    tickVoice();
  };
  pause=function(){
    if(song)basePause();else{playing=false;cancelAnimationFrame(radioVoiceRaf);iconUse($('#playUse'),'#i-play')}
    radioVoice.audio?.pause();
    updateTransport();
  };
  stop=function(){
    baseStop();playing=false;cancelAnimationFrame(radioVoiceRaf);radioVoice.audio?.pause();updateTransport();
  };
  updateTransport=function(){
    baseUpdate();
    const sec=radioTimeFromBeat();
    if(radioVoice.duration){
      $('#timeText').textContent=`${fmtTime(sec)} / ${fmtTime(radioVoice.duration)}`;
      redrawTimedSketch(sec);
    }
    drawRadioCursor();
  };
  $('#playBtn').onclick=()=>{if(playing)pause();else{if(currentBeat>=48)currentBeat=0;play()}};
  $('#scrub').oninput=e=>{
    const was=playing;if(was)pause();
    currentBeat=Number(e.target.value);
    if(radioVoice.audio)radioVoice.audio.currentTime=radioTimeFromBeat();
    updateTransport();
    if(was)play();
  };
}

function radioTheoryPrompt(revise=false){
  const prev=revise?JSON.stringify(pipeline):'(none)';
  const change=$('#radioPrompt')?.value.trim()||$('#changeInput').value.trim();
  const sound=radioVoice.reading||voiceSummary();
  const mode=RADIO_MODE_INSTRUCTIONS[radioMode];
  return `You are the scoring intelligence inside SKETCHRADIO: WORD + VOICE + TIMED DRAWING + WORLD + SCORE.
Populate the JSON schema only. Your job is to expose and operate relations, not decorate a poem with generic background music.

SCORING MODE: ${radioMode}
${mode}

POEM / WORD:
${$('#wordInput').value.trim()||'(none)'}

VOICE / RADIO EVIDENCE:
${sound}

TIMED DRAWING:
${timedSketchSummary()}

USER REQUEST:
${change||'(none)'}

PREVIOUS STATE:
${prev}

AGENT 1 — LISTEN + PROPOSE THEORY
Return 3-4 provisional causal rules. Sources can include poem cadence, phrase ending, silence, voice pressure, voice envelope, timed stroke direction, stroke onset, mark density, world/spectrum region. Targets are musical operations such as rhythm, silence, register, pitch contour, timbre, density, duration, section contrast. State the evidence and audible consequence. Never pretend a visual proxy gives exact linguistic or pitch facts.

AGENT 2 — SCORE THE RELATION
Create A1 / B / A2 as a 12-bar test of those rules. A1 establishes a relation. B changes or resists one rule. A2 returns with memory. If voice is present, leave room for it.

AGENT 3 — PLAYABLE EVENTS
Make practical MIDI note events. Keep motifs legible. Use rests deliberately. Do not fill every gap.

AGENT 4 — ASSEMBLE
Build the playable score. Preserve causal evidence from poem, voice-map and timed marks.

The current sound-map and sketch images are supplied as visual evidence. Treat the user's prompt as an instruction to CHANGE THE RELATION.`;
}
async function radioMakeOrLoop(){
  if(!verified){$('#gate').classList.remove('hidden');return}
  const source=$('#wordInput').value.trim()||baseHasSketchInk()||radioVoice.buffer;
  if(!song&&!source)return;
  stop();
  const previousSong=song,previousPipeline=pipeline;
  busy(true,song?'RESHAPING':'SCORING');
  try{
    const images=[];
    if(baseHasSketchInk())images.push({label:'CURRENT SKETCH / TIMED MARK RESULT',url:canvasDataURL(sketch)});
    if(radioVoice.buffer){
      radioRenderWorld();
      images.push({label:'CURRENT VOICE / RADIO SOUND MAP',url:canvasDataURL(specBase)});
    }
    if(dirty.spectrum)images.push({label:'CURRENT WORLD / SOUND MAP WITH USER MARKS',url:canvasDataURL(specInk,specBase)});
    const promptText=radioTheoryPrompt(!!song);
    pipeline=await structured(contentForImages(promptText,images),'sketchradio_pipeline',THEORY_PIPELINE_SCHEMA);
    song=pipeline.agent4;
    if(radioVoice.duration){
      const syncTempo=Math.max(40,Math.min(220,2880/radioVoice.duration));
      song.tempo_bpm=Math.round(syncTempo);
      pipeline.agent1.tempo_bpm=song.tempo_bpm;
    }
    await revealBuiltTheory(pipeline);
    await renderSong();
    recordGeneration?.(previousSong,previousPipeline,promptText);
    resetDirty();
    $('#radioPrompt').value='';
    radioRenderWorld();
    setState(`READY · ${radioMode} · TIME IS REAL`,'good');
    $('#makeLabel').textContent='RESHAPE';
    if(innerWidth<=900)goColumn(2);
  }catch(e){setState(e.message,'bad')}
  finally{busy(false);updateMakeState()}
}
function installRadioAIPatches(){
  const baseGuide=guideContextPrompt;
  guideContextPrompt=function(mode,q){
    return `${baseGuide(mode,q)}
CURRENT VOICE / RADIO: ${radioVoice.reading||voiceSummary()}
CURRENT TIMED DRAWING: ${timedSketchSummary()}
CURRENT SCORING MODE: ${radioMode}
Treat POEM, VOICE, DRAWING, WORLD and SCORE as one temporal field. If asked to change something, name the relation to change and an audible/visible consequence.`;
  };
  const baseRender=renderSpectrum;
  renderSpectrum=async function(){
    if(!radioVoice.buffer)return baseRender();
    radioRenderWorld();
  };
  const baseUpdateMake=updateMakeState;
  updateMakeState=function(){
    baseUpdateMake();
    const hasSource=$('#wordInput').value.trim()||baseHasSketchInk()||radioVoice.buffer;
    $('#makeBtn').disabled=!verified||(!song&&!hasSource);
    $('#makeLabel').textContent=song?'RESHAPE':'SCORE';
  };
  $('#makeBtn').onclick=radioMakeOrLoop;
}

const baseHasSketchInk=hasSketchInk;
installRadioSurface();
