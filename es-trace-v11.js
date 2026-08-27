(()=>{
"use strict";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)], clamp=(v,a,b)=>Math.max(a,Math.min(b,v)), copy=o=>JSON.parse(JSON.stringify(o));
const hash=s=>{let h=2166136261;for(let c of s){h^=c.charCodeAt(0);h=Math.imul(h,16777619)}return h>>>0};
const rng=seed=>{let x=seed||1;return()=>((x^=x<<13,x^=x>>>17,x^=x<<5)>>>0)/4294967296};
const EARS="https://earsketch.gatech.edu/backend-static/";
const S={
  ctx:null,master:null,playing:false,nodes:[],catalog:[],catalogReady:false,
  sounds:{A:null,B:null},selected:"GROOVE",scope:"GROOVE",
  groove:null,before:null,beforeSounds:null,candidate:null,candidateMeta:null,history:[],draft:null,shapeDraft:null,selectedHit:0,
  view:"listen",findMode:"near",
  ai:{key:"",model:"",ready:false}
};

function status(k,t,b=false){$("#statusKey").textContent=k;$("#statusText").textContent=t;$("#pulse").classList.toggle("busy",b);$("#progress").classList.toggle("busy",b)}
function openSheet(id){$("#"+id).classList.add("open")}function closeSheet(id){$("#"+id).classList.remove("open")}
$$("[data-close]").forEach(b=>b.onclick=()=>closeSheet(b.dataset.close));

function stepForView(v){return v==="listen"?"listen":v==="compare"?"compare":v==="keep"?"keep":"change"}
function setView(v,op){
  S.view=v;$$('.view').forEach(x=>x.classList.remove('active'));$('#'+v+'View').classList.add('active');
  $$('[data-step]').forEach(b=>b.classList.toggle('on',b.dataset.step===stepForView(v)));
  $('#crumbOp').textContent=(op||v).toUpperCase();
  updateCrumb();
}
function updateCrumb(){
  $('#crumbGroove').textContent=S.groove?.name||'GROOVE';
  $('#crumbObject').textContent=S.selected==='GROOVE'?'GROOVE':`SOUND ${S.selected}`;
  $('#scopeName').textContent=S.scope==='GROOVE'?'THE GROOVE':S.scope==='RELATION'?'A ↔ B':`SOUND ${S.scope}`;
}
function selectObject(o){
  S.selected=o;if(o==='GROOVE')S.scope='GROOVE';else S.scope=o;
  $$('.objectCard').forEach(b=>b.classList.toggle('sel',b.dataset.object===o));
  updateCrumb();updateChangeCopy()
}
function updateChangeCopy(){
  let name=S.selected==='GROOVE'?'THE GROOVE':`SOUND ${S.selected}`;
  $('#changeSub').textContent=`Working on: ${name}. Selection alone changes nothing.`;
  $('#timingSub').textContent=`Working on: ${name}. Drag the hits. Nothing is committed yet.`;
  $('#shapeSub').textContent=`Working on: ${name}`;
}
async function ensureCtx(){
  if(!S.ctx)S.ctx=new (window.AudioContext||window.webkitAudioContext)();
  await S.ctx.resume();
  if(!S.master){S.master=S.ctx.createGain();S.master.gain.value=.82;S.master.connect(S.ctx.destination)}
  return S.ctx
}
function seedBuffer(kind){
  let c=S.ctx,sr=c.sampleRate,len=Math.floor(sr*.38),b=c.createBuffer(1,len,sr),d=b.getChannelData(0);
  if(kind==='kick'){let ph=0;for(let i=0;i<len;i++){let t=i/sr,f=115*Math.pow(42/115,t/.22);ph+=2*Math.PI*f/sr;d[i]=Math.sin(ph)*Math.exp(-t*14)}}
  else{let r=rng(77);for(let i=0;i<len;i++){let t=i/sr;d[i]=(r()*2-1)*Math.exp(-t*18)*.45+Math.sin(2*Math.PI*900*t)*Math.exp(-t*16)*.4}}
  return b
}
async function initAudio(){
  await ensureCtx();
  S.sounds.A={id:'seed-kick',name:'SEED KICK',kind:'LOCAL',buffer:seedBuffer('kick'),shape:{start:0,end:1,rate:1,gain:1,reverse:false}};
}
async function loadCatalog(){
  try{let r=await fetch(EARS+'audio-standard_4.json');if(!r.ok)throw Error(r.status);S.catalog=(await r.json()).filter(x=>x&&x.name&&x.path);S.catalogReady=true;$('#catalogState').textContent=`ES ${S.catalog.length}`;status('READY','EarSketch sounds ready')}
  catch(e){$('#catalogState').textContent='ES OFF'}
}
async function decodeRemote(meta){
  await ensureCtx();status('LOAD',meta.name,true);let r=await fetch(EARS+meta.path);if(!r.ok)throw Error(`sample ${r.status}`);let a=await r.arrayBuffer(),b=await S.ctx.decodeAudioData(a);status('READY',meta.name);return{id:meta.name,name:meta.name,kind:'EARSKETCH',meta,buffer:b,shape:{start:0,end:1,rate:1,gain:1,reverse:false}}
}
async function importSound(file){await ensureCtx();let a=await file.arrayBuffer(),b=await S.ctx.decodeAudioData(a);return{id:`local-${Date.now()}`,name:file.name,kind:'IMPORT',buffer:b,shape:{start:0,end:1,rate:1,gain:1,reverse:false}}}

function defaultGroove(){return{id:`g-${Date.now()}`,name:'GROOVE 01',bpm:116,beats:8,events:[{slot:'A',beat:0,gain:1,rate:1},{slot:'A',beat:5.25,gain:.76,rate:1}],note:'two hits with a large hole'}}
function normalize(g){g.events=g.events.filter(e=>S.sounds[e.slot]&&e.beat>=0&&e.beat<g.beats).sort((a,b)=>a.beat-b.beat);return g}
function metrics(g){
  let e=[...g.events].sort((a,b)=>a.beat-b.beat),cells=new Set(e.map(x=>Math.floor(x.beat*4))).size,total=g.beats*4,gaps=[];
  for(let i=0;i<e.length;i++){let end=i===e.length-1?g.beats:e[i+1].beat;gaps.push(end-e[i].beat)}
  let off=e.filter(x=>Math.abs(x.beat*2-Math.round(x.beat*2))>.04).length;
  return{events:e.length,silence:1-cells/total,maxGap:gaps.length?Math.max(...gaps):g.beats,offgrid:e.length?off/e.length:0}
}
function fmtMetric(v){return Number(v.toFixed(2)).toString()}
function sampleValue(obj,frac){
  let b=obj.buffer,d=b.getChannelData(0),sh=obj.shape||{start:0,end:1,rate:1,gain:1,reverse:false},a=Math.floor(sh.start*d.length),z=Math.max(a+1,Math.floor(sh.end*d.length)),span=z-a,p=clamp(frac,0,.999999)*span,i=Math.floor(p),f=p-i,idx=sh.reverse?z-1-i:a+i,idx2=sh.reverse?Math.max(a,idx-1):Math.min(z-1,idx+1);
  return ((d[idx]||0)*(1-f)+(d[idx2]||0)*f)*(sh.gain||1)
}
function renderPCM(g,sounds=S.sounds){
  let sr=16000,dur=g.beats*60/g.bpm+.5,n=Math.ceil(dur*sr),out=new Float32Array(n);
  for(let ev of g.events){let o=sounds[ev.slot];if(!o?.buffer)continue;let sh=o.shape,start=Math.floor(ev.beat*60/g.bpm*sr),srcDur=(sh.end-sh.start)*o.buffer.duration/(sh.rate*(ev.rate||1)),count=Math.min(n-start,Math.floor(srcDur*sr));
    for(let i=0;i<count;i++){let frac=i/Math.max(1,count);out[start+i]+=sampleValue(o,frac)*(ev.gain||1)}
  }
  let peak=0;for(let v of out)peak=Math.max(peak,Math.abs(v));if(peak>1){let k=.96/peak;for(let i=0;i<out.length;i++)out[i]*=k}
  return{pcm:out,sr}
}
function analyzePCM(g,sounds=S.sounds){
  let r=renderPCM(g,sounds),base=metrics(g),pcm=r.pcm,win=256,hop=128,rms=[];
  for(let i=0;i+win<pcm.length;i+=hop){let s=0;for(let j=0;j<win;j++)s+=pcm[i+j]*pcm[i+j];rms.push(Math.sqrt(s/win))}
  let mx=Math.max(.00001,...rms),silent=rms.filter(x=>x<mx*.06).length/Math.max(1,rms.length);
  return{...base,silence:Math.max(base.silence,silent)}
}
function metricHTML(k,v){return`<div class="metric"><small>${k}</small><b>${v}</b></div>`}
function canvasFit(c,h){let dpr=devicePixelRatio||1,w=c.clientWidth;c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);let x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);return{x,w,h}}
function drawGroove(c,g,selectedHit=-1,editable=false){
  let {x,w,h}=canvasFit(c,c=== $('#timingCanvas')?270:210);x.fillStyle='#fff';x.fillRect(0,0,w,h);
  x.strokeStyle='rgba(0,0,0,.16)';x.lineWidth=1;for(let b=0;b<=g.beats;b++){let px=b/g.beats*w;x.beginPath();x.moveTo(px,0);x.lineTo(px,h);x.stroke()}
  let render=renderPCM(g),pcm=render.pcm;x.strokeStyle='#000';x.lineWidth=1;x.beginPath();for(let px=0;px<w;px++){let i=Math.floor(px/w*pcm.length),y=h/2-pcm[i]*h*.25;if(px===0)x.moveTo(px,y);else x.lineTo(px,y)}x.stroke();
  g.events.forEach((e,i)=>{let px=e.beat/g.beats*w,y=e.slot==='A'?h*.32:h*.68;x.fillStyle=e.slot==='A'?'#000':'#efff00';x.strokeStyle='#000';x.lineWidth=i===selectedHit?4:2;x.beginPath();x.arc(px,y,editable?16:13,0,Math.PI*2);x.fill();x.stroke()});
}
function renderMain(){
  let m=analyzePCM(S.groove),hasB=!!S.sounds.B;
  $('#grooveTitle').textContent=S.groove.name;$('#grooveSub').textContent=`2 bars · ${hasB?'two sounds':'one sound'} · room to breathe`;
  $('#listenSummary').textContent=`${m.events} events · ${Math.round(m.silence*100)}% silence`;
  $('#metrics').innerHTML=metricHTML('EVENTS',m.events)+metricHTML('SILENCE',`${Math.round(m.silence*100)}%`)+metricHTML('MAX GAP',`${fmtMetric(m.maxGap)}b`)+metricHTML('OFF-GRID',`${Math.round(m.offgrid*100)}%`);
  $('#aName').textContent=`SOUND A · ${S.sounds.A?.name||'EMPTY'}`;$('#aState').textContent=`${S.groove.events.filter(e=>e.slot==='A').length} hits · ${S.sounds.A?.kind||''}`;
  $('#bName').textContent=`SOUND B · ${S.sounds.B?.name||'EMPTY'}`;$('#bState').textContent=S.sounds.B?`${S.groove.events.filter(e=>e.slot==='B').length} hits · ${S.sounds.B.kind}`:'add only if useful';
  $(".objectCard[data-object='B']").classList.toggle('empty',!S.sounds.B);
  drawGroove($('#grooveCanvas'),S.groove);
  updateCrumb()
}
function reverseBuffer(obj){
  if(obj._rev)return obj._rev;let b=obj.buffer,r=S.ctx.createBuffer(b.numberOfChannels,b.length,b.sampleRate);for(let ch=0;ch<b.numberOfChannels;ch++){let a=b.getChannelData(ch),d=r.getChannelData(ch);for(let i=0;i<a.length;i++)d[i]=a[a.length-1-i]}obj._rev=r;return r
}
function scheduleObject(o,when,ev){
  let sh=o.shape,src=S.ctx.createBufferSource(),g=S.ctx.createGain(),buf=sh.reverse?reverseBuffer(o):o.buffer;src.buffer=buf;
  let start=sh.reverse?(1-sh.end)*buf.duration:sh.start*buf.duration,end=sh.reverse?(1-sh.start)*buf.duration:sh.end*buf.duration,dur=Math.max(.01,end-start);
  src.playbackRate.value=sh.rate*(ev.rate||1);g.gain.value=(sh.gain||1)*(ev.gain||1);src.connect(g).connect(S.master);src.start(when,start,Math.min(dur,buf.duration-start));S.nodes.push(src,g)
}
async function play(g=S.groove,sounds=S.sounds){
  if(S.playing)stop();await ensureCtx();S.playing=true;$('#dockListen').textContent='■ STOP';$('#listenBtn').textContent='■ STOP';status('PLAY','listening',true);let st=S.ctx.currentTime+.04,spb=60/g.bpm;
  for(let e of g.events){let o=sounds[e.slot];if(o?.buffer)scheduleObject(o,st+e.beat*spb,e)}
  setTimeout(()=>{if(S.playing)stop()},(g.beats*spb+.4)*1000)
}
function stop(){if(!S.playing)return;S.playing=false;S.nodes.forEach(n=>{try{n.disconnect?.()}catch{}});S.nodes=[];$('#dockListen').textContent='▶ LISTEN';$('#listenBtn').textContent='▶ LISTEN';status('READY','listen → change → compare → keep')}

function bindBaseActions(){
  $('#actionGrid').innerHTML=`<button data-action="timing"><b>TIMING</b><small>move hits, remove hits, create a pause</small></button>
  <button data-action="sound"><b>SOUND</b><small>find another sound or shape this one</small></button>
  <button data-action="space"><b>SPACE</b><small>make the groove thinner or leave a larger hole</small></button>
  <button data-action="variations"><b>VARIATIONS</b><small>hear four different possibilities</small></button>`;
  $$('[data-action]').forEach(b=>b.onclick=()=>handleAction(b.dataset.action));
}
function beginChange(){stop();bindBaseActions();setView('change','change');updateChangeCopy()}
function beginTiming(){
  S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);S.draft=copy(S.groove);S.selectedHit=Math.max(0,S.draft.events.findIndex(e=>S.selected==='GROOVE'||e.slot===S.selected));drawGroove($('#timingCanvas'),S.draft,S.selectedHit,true);setView('timing','timing')
}
function chooseSpace(){
  S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);let c=copy(S.groove),filtered=[];
  for(let i=0;i<c.events.length;i++)if(i===0||i===c.events.length-1||i%2===0)filtered.push(c.events[i]);
  if(filtered.length===c.events.length&&filtered.length>1)filtered.splice(1,1);c.events=filtered;prepareCompare(c,{label:'MORE SPACE',sounds:copySoundRefs(S.sounds)})
}
function cloneSound(o){return o?{...o,shape:{...o.shape}}:null}
function copySoundRefs(s){return{A:cloneSound(s.A),B:cloneSound(s.B)}}
function mutate(g,type,seed){
  let r=rng(seed),n=copy(g),e=n.events;
  if(type==='SPARSER'){e=e.filter((x,i)=>i===0||i===e.length-1||r()>.45);if(e.length>4)e=e.slice(0,4)}
  if(type==='LATER'){if(e[1])e[1].beat=clamp(e[1].beat+.5,0,n.beats-.1)}
  if(type==='CROOKED'){e.forEach((x,i)=>{if(i&&r()>.3)x.beat=clamp(x.beat+(r()>.5?.125:-.125),0,n.beats-.1)})}
  if(type==='EMPTY'){e=e.filter((x,i)=>i===0||i===e.length-1);if(e.length>3)e=e.slice(0,3)}
  n.events=e;return normalize(n)
}
function showVariations(){
  let types=['SPARSER','LATER','CROOKED','ALMOST EMPTY'],mapType={'SPARSER':'SPARSER','LATER':'LATER','CROOKED':'CROOKED','ALMOST EMPTY':'EMPTY'},h=$('#variationGrid');h.innerHTML='';
  types.forEach((name,i)=>{let g=mutate(S.groove,mapType[name],hash(S.groove.id+name+i)),m=analyzePCM(g),b=document.createElement('button');b.className='variation';b.innerHTML=`<b>${name}</b><small>${m.events} events · ${Math.round(m.silence*100)}% silence<br>max gap ${fmtMetric(m.maxGap)} beats</small><div class="miniPattern"></div>`;let p=b.querySelector('.miniPattern');g.events.forEach(e=>{let x=document.createElement('i');x.className='miniHit';x.style.left=`${e.beat/g.beats*100}%`;x.style.height=e.slot==='A'?'14px':'8px';p.append(x)});b.onclick=()=>{S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);play(g);setTimeout(()=>prepareCompare(g,{label:name,sounds:copySoundRefs(S.sounds)}),250)};h.append(b)});
  setView('variations','variations')
}
function prepareCompare(candidate,meta={}){
  stop();
  if(!S.beforeSounds)S.beforeSounds=copySoundRefs(S.sounds);
  S.candidate=normalize(copy(candidate));S.candidateMeta=meta;
  setView('compare','compare');renderCompare()
}
function renderCompare(){
  let a=analyzePCM(S.before||S.groove,S.beforeSounds||S.sounds),
      b=analyzePCM(S.candidate,S.candidateMeta?.sounds||S.sounds),
      label=S.candidateMeta?.label||'CHANGE';
  $('#compareTitle').textContent=`COMPARE · ${label}`;$('#compareSub').textContent='Hear before. Hear after. Then decide.';
  drawGroove($('#compareCanvas'),S.candidate);
  let rows=[['EVENTS',a.events,b.events],['SILENCE',`${Math.round(a.silence*100)}%`,`${Math.round(b.silence*100)}%`],['MAX GAP',`${fmtMetric(a.maxGap)}b`,`${fmtMetric(b.maxGap)}b`],['OFF-GRID',`${Math.round(a.offgrid*100)}%`,`${Math.round(b.offgrid*100)}%`]];
  $('#delta').innerHTML=rows.map(r=>`<div class="deltaRow"><span><b>${r[0]}</b></span><span>${r[1]}</span><span>${r[2]}</span></div>`).join('')
}
function keepCandidate(){
  if(!S.candidate)return;S.history.unshift({groove:copy(S.groove),sounds:copySoundRefs(S.sounds),time:Date.now()});
  if(S.candidateMeta?.sounds)S.sounds=S.candidateMeta.sounds;
  S.groove=copy(S.candidate);let n=S.history.length+1;S.groove.name=`GROOVE ${String(n).padStart(2,'0')}`;
  S.before=S.beforeSounds=S.candidate=null;S.candidateMeta=null;S.shapeDraft=null;
  renderHistory();renderMain();setView('listen','keep');$$('[data-step]').forEach(b=>b.classList.toggle('on',b.dataset.step==='keep'));$('#crumbOp').textContent='KEPT';status('KEPT','change became current');setTimeout(()=>{setView('listen','listen')},800)
}
function undoCandidate(){S.before=S.beforeSounds=S.candidate=null;S.candidateMeta=null;S.shapeDraft=null;setView('listen','listen');renderMain();status('UNDO','current groove unchanged')}
function renderHistory(){
  let h=$('#historyList');h.innerHTML='';if(!S.history.length){h.innerHTML='<div class="stateSub">No kept ancestors yet.</div>';return}
  S.history.forEach((x,i)=>{let m=analyzePCM(x.groove,x.sounds),b=document.createElement('button');b.className='hist';b.innerHTML=`<b>${x.groove.name}</b><small>${m.events} events · ${Math.round(m.silence*100)}% silence · ${new Date(x.time).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</small>`;b.onclick=()=>{S.history.unshift({groove:copy(S.groove),sounds:copySoundRefs(S.sounds),time:Date.now()});S.groove=copy(x.groove);S.sounds=x.sounds;renderMain();renderHistory();setView('listen','listen');status('HISTORY',`restored ${S.groove.name}`)};h.append(b)})
}

function finderScore(meta,q,mode){
  let text=[meta.name,meta.folder,meta.instrument,meta.genre,meta.artist].filter(Boolean).join(' ').toLowerCase(),words=q.toLowerCase().split(/[^a-z0-9]+/).filter(x=>x.length>2),s=0;
  for(let w of words)if(text.includes(w))s+=6;
  let current=S.sounds[S.selected==='GROOVE'?'A':S.selected],name=current?.name?.toLowerCase()||'';
  if(mode==='near')for(let w of name.split(/[_\s]+/))if(w.length>3&&text.includes(w))s+=4;
  if(mode==='attack'&&/drum|perc|hit|kick|snare|hat|clap|pluck|mallet/.test(text))s+=7;
  if(mode==='lineage'&&/afro|gnawa|folk|jazz|world|latin|indian|arab|asia|electronic/.test(text))s+=4;
  if(mode==='strange')s+=(hash(meta.name+name)%100)/10;
  return s
}
function renderResults(){
  let q=$('#soundSearch').value.trim(),list=S.catalog.map(m=>[finderScore(m,q,S.findMode),m]).filter(x=>q?x[0]>0:true).sort((a,b)=>b[0]-a[0]).slice(0,80),h=$('#results');h.innerHTML='';
  $('#resultCount').textContent=S.catalogReady?`${list.length} results · ${S.findMode.toUpperCase()}`:'catalog loading…';
  for(let [,m] of list){let b=document.createElement('button');b.className='result';b.innerHTML=`<b>${m.name}</b><small>${m.folder||m.instrument||m.genre||'EarSketch'}</small>`;b.onclick=async()=>{let slot=S.selected==='GROOVE'?(S.sounds.B?'A':'B'):S.selected;try{let o=await decodeRemote(m),sounds=copySoundRefs(S.sounds);sounds[slot]=o;let c=copy(S.groove);if(!c.events.some(e=>e.slot===slot))c.events.push({slot,beat:3.5,gain:.7,rate:1});S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);prepareCompare(c,{label:`SOUND ${slot} → ${m.name}`,sounds})}catch(e){status('FAILED',e.message)}};h.append(b)}
}
function openFinder(){
  let slot=S.selected==='GROOVE'?(S.sounds.B?'A':'B'):S.selected;$('#finderTarget').textContent=`SOUND ${slot}`;$('#soundSearch').value='';renderResults();setView('finder','find sound')
}
function openShape(){
  let slot=S.selected==='GROOVE'?'A':S.selected,o=S.sounds[slot];if(!o){openFinder();return}
  S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);S.shapeDraft=cloneSound(o);
  S.candidateMeta={label:`SHAPE SOUND ${slot}`,shapeSlot:slot};
  $('#shapeSub').textContent=`Working on: SOUND ${slot} · ${o.name}`;
  $('#shapeStart').value=S.shapeDraft.shape.start*100;$('#shapeEnd').value=S.shapeDraft.shape.end*100;$('#shapeRate').value=S.shapeDraft.shape.rate*100;$('#shapeGain').value=S.shapeDraft.shape.gain*100;
  $('#reverseBtn').textContent=S.shapeDraft.shape.reverse?'REVERSED':'REVERSE';drawShape();setView('shape','shape sound')
}
function drawShape(){
  let o=S.shapeDraft,c=$('#shapeCanvas'),{x,w,h}=canvasFit(c,200);x.fillStyle='#fff';x.fillRect(0,0,w,h);if(!o)return;
  let d=o.buffer.getChannelData(0);x.strokeStyle='#000';x.lineWidth=1;x.beginPath();
  for(let px=0;px<w;px++){let i=Math.floor(px/w*d.length),y=h/2-d[i]*h*.42;if(px===0)x.moveTo(px,y);else x.lineTo(px,y)}x.stroke();
  x.fillStyle='rgba(239,255,0,.5)';x.fillRect(o.shape.start*w,0,(o.shape.end-o.shape.start)*w,h);x.strokeStyle='#000';x.strokeRect(o.shape.start*w,0,(o.shape.end-o.shape.start)*w,h)
}
function syncShape(){
  let o=S.shapeDraft;if(!o)return;
  o.shape.start=+$('#shapeStart').value/100;o.shape.end=Math.max(o.shape.start+.03,+$('#shapeEnd').value/100);o.shape.rate=+$('#shapeRate').value/100;o.shape.gain=+$('#shapeGain').value/100;drawShape()
}
async function hearShape(){if(!S.shapeDraft)return;await ensureCtx();scheduleObject(S.shapeDraft,S.ctx.currentTime+.03,{rate:1,gain:1})}
function cancelShape(){S.shapeDraft=null;S.before=null;S.beforeSounds=null;S.candidateMeta=null;beginChange()}
function compareShape(){
  let slot=S.candidateMeta.shapeSlot,sounds=copySoundRefs(S.sounds);sounds[slot]=cloneSound(S.shapeDraft);
  prepareCompare(S.groove,{label:`SHAPE SOUND ${slot}`,sounds})
}
function timingPointer(){
  let c=$('#timingCanvas'),drag=-1;
  c.onpointerdown=e=>{let r=c.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,w=r.width,h=r.height,best=-1,dist=999;
    S.draft.events.forEach((ev,i)=>{if(S.selected!=='GROOVE'&&ev.slot!==S.selected)return;let px=ev.beat/S.draft.beats*w,py=ev.slot==='A'?h*.32:h*.68,d=Math.hypot(px-x,py-y);if(d<dist){dist=d;best=i}});
    if(best>=0&&dist<42){drag=best;S.selectedHit=best;c.setPointerCapture(e.pointerId);drawGroove(c,S.draft,S.selectedHit,true)}
  };
  c.onpointermove=e=>{if(drag<0)return;let r=c.getBoundingClientRect(),beat=clamp((e.clientX-r.left)/r.width*S.draft.beats,0,S.draft.beats-.05);S.draft.events[drag].beat=Math.round(beat*8)/8;normalize(S.draft);S.selectedHit=S.draft.events.indexOf(S.draft.events.find(x=>x===S.draft.events[drag]))>=0?drag:0;drawGroove(c,S.draft,S.selectedHit,true)};
  c.onpointerup=e=>{drag=-1}
}
function chatLocal(text){
  let q=text.toLowerCase(),c=copy(S.groove),label='CHAT';
  if(/empty|space|silence|breathe|less/.test(q)){c=mutate(c,'SPARSER',hash(text));label='MORE SPACE'}
  else if(/late|later|drag/.test(q)){c=mutate(c,'LATER',hash(text));label='LATER'}
  else if(/crooked|swing|early|push|unstable/.test(q)){c=mutate(c,'CROOKED',hash(text));label='CROOKED'}
  else if(/second sound|answer|reply|call/.test(q)&&S.sounds.B){if(!c.events.some(e=>e.slot==='B'))c.events.push({slot:'B',beat:3.5,gain:.7,rate:1});label='B ANSWERS A'}
  else{c=mutate(c,'SPARSER',hash(text));label='SIMPLIFY'}
  S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);prepareCompare(c,{label,sounds:copySoundRefs(S.sounds)})
}
async function api(path,opt={}){let r=await fetch('https://api.openai.com/v1'+path,{...opt,headers:{Authorization:`Bearer ${S.ai.key}`,'Content-Type':'application/json'}}),txt=await r.text(),d;try{d=JSON.parse(txt)}catch{d={raw:txt}}if(!r.ok)throw Error(d?.error?.message||`HTTP ${r.status}`);return d}
function outText(d){if(typeof d?.output_text==='string')return d.output_text;let a=[];for(let i of d?.output||[])for(let c of i.content||[])if(typeof c?.text==='string')a.push(c.text);return a.join('\n')}
async function connectAI(){
  S.ai.key=$('#apiKey').value.trim();if(!S.ai.key)return;status('OPENAI','checking',true);
  try{let d=await api('/models'),ids=(d.data||[]).map(x=>x.id).filter(x=>/^gpt-5/.test(x)&&!/(audio|realtime|image|codex|transcribe)/i.test(x));S.ai.model=ids.find(x=>x.includes('5.6-sol'))||ids[0];if(!S.ai.model)throw Error('no compatible model');let p=await api('/responses',{method:'POST',body:JSON.stringify({model:S.ai.model,input:'Reply OK',max_output_tokens:64,store:false})});if(!/OK/i.test(outText(p)))throw Error('ping failed');S.ai.ready=true;$('#aiBtn').classList.add('ready');$('#aiBtn').textContent='AI·ON';$('#apiMsg').textContent=`READY · ${S.ai.model}`;status('READY','AI connected');closeSheet('aiSheet')}catch(e){$('#apiMsg').textContent=e.message;status('AI ERROR',e.message)}
}
async function chatAI(text){
  let m=analyzePCM(S.groove),scope=S.scope,req={model:S.ai.model,input:`You operate on a sparse 2-bar groove. The user is talking to ${scope}. Preserve clarity and negative space. Never exceed 7 events. Return JSON only with label and events.
Instruction: ${text}
Current groove: ${JSON.stringify(S.groove)}
Measured render: ${JSON.stringify(m)}
Sound B exists: ${!!S.sounds.B}`,max_output_tokens:900,store:false,text:{format:{type:'json_schema',name:'groove_change',strict:true,schema:{type:'object',properties:{label:{type:'string'},events:{type:'array',maxItems:7,items:{type:'object',properties:{slot:{type:'string',enum:['A','B']},beat:{type:'number',minimum:0,maximum:7.99},gain:{type:'number',minimum:.05,maximum:1.5},rate:{type:'number',minimum:.25,maximum:3}},required:['slot','beat','gain','rate'],additionalProperties:false}}},required:['label','events'],additionalProperties:false}}}};
  status('OPENAI','proposing one change',true);
  try{let d=await api('/responses',{method:'POST',body:JSON.stringify(req)}),o=JSON.parse(outText(d)),c=copy(S.groove);c.events=o.events.filter(e=>S.sounds[e.slot]);S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);prepareCompare(c,{label:o.label,sounds:copySoundRefs(S.sounds)});status('COMPARE','before / after ready')}
  catch(e){status('AI→LOCAL',e.message);chatLocal(text)}
}
function doChat(){let t=$('#chatInput').value.trim();if(!t)return;S.ai.ready?chatAI(t):chatLocal(t)}

$$('.objectCard').forEach(b=>b.onclick=()=>selectObject(b.dataset.object));
$('#listenBtn').onclick=()=>play();$('#dockListen').onclick=()=>play();$('#changeBtn').onclick=beginChange;$('#dockChange').onclick=beginChange;$('#changeBack').onclick=()=>setView('listen','listen');
function handleAction(a){
  if(a==='timing')return beginTiming();
  if(a==='space')return chooseSpace();
  if(a==='variations')return showVariations();
  if(a==='sound'){
    if(S.selected==='GROOVE'&&!S.sounds.B)return openFinder();
    $('#changeTitle').textContent='WHAT DO YOU WANT TO DO WITH THE SOUND?';
    $('#changeSub').textContent=`Working on: ${S.selected==='GROOVE'?'THE GROOVE':`SOUND ${S.selected}`}`;
    $('#actionGrid').innerHTML=`<button id="findSoundNow"><b>FIND ANOTHER</b><small>search the live sound catalog</small></button>
      <button id="shapeNow"><b>SHAPE THIS</b><small>trim, reverse, change rate or gain</small></button>
      <button id="removeNow"><b>REMOVE</b><small>remove the selected sound from the groove</small></button>
      <button id="soundBack"><b>BACK</b><small>choose another kind of change</small></button>`;
    $('#findSoundNow').onclick=openFinder;$('#shapeNow').onclick=openShape;
    $('#removeNow').onclick=()=>{
      let slot=S.selected==='GROOVE'?'B':S.selected;
      if(slot==='A'){status('KEEP A','Sound A anchors the current groove');return}
      let c=copy(S.groove);c.events=c.events.filter(e=>e.slot!==slot);let sounds=copySoundRefs(S.sounds);sounds[slot]=null;
      S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);prepareCompare(c,{label:`REMOVE SOUND ${slot}`,sounds})
    };
    $('#soundBack').onclick=beginChange;
  }
}
bindBaseActions();
$('#scopeBtn').onclick=()=>{let opts=S.sounds.B?['GROOVE','A','B','RELATION']:['GROOVE','A'],i=opts.indexOf(S.scope);S.scope=opts[(i+1)%opts.length];updateCrumb()};
$('#chatGo').onclick=doChat;$('#chatInput').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();doChat()}};
$('#lessBtn').onclick=()=>{if(S.draft.events.length<=1)return;let i=clamp(S.selectedHit,0,S.draft.events.length-1);S.draft.events.splice(i,1);S.selectedHit=clamp(i-1,0,S.draft.events.length-1);drawGroove($('#timingCanvas'),S.draft,S.selectedHit,true)};
$('#addHitBtn').onclick=()=>{if(S.draft.events.length>=7)return;let slot=S.selected==='GROOVE'?(S.sounds.B?'B':'A'):S.selected,spots=[1.5,3.5,6.5,2.75,7.25],beat=spots.find(x=>!S.draft.events.some(e=>Math.abs(e.beat-x)<.3));if(beat!=null){S.draft.events.push({slot,beat,gain:.65,rate:1});normalize(S.draft);S.selectedHit=S.draft.events.length-1;drawGroove($('#timingCanvas'),S.draft,S.selectedHit,true)}};
$('#timingCompare').onclick=()=>prepareCompare(S.draft,{label:'TIMING',sounds:copySoundRefs(S.sounds)});$('#timingCancel').onclick=()=>{S.draft=null;setView('change','change')};
$('#variationBack').onclick=()=>beginChange();$('#beforeBtn').onclick=()=>play(S.before||S.groove,S.beforeSounds||S.sounds);$('#afterBtn').onclick=()=>play(S.candidate,S.candidateMeta?.sounds||S.sounds);$('#keepBtn').onclick=keepCandidate;$('#undoBtn').onclick=undoCandidate;$('#tryAgainBtn').onclick=beginChange;
$$('[data-find]').forEach(b=>b.onclick=()=>{S.findMode=b.dataset.find;$$('[data-find]').forEach(x=>x.classList.toggle('on',x===b));renderResults()});$('#soundSearch').oninput=renderResults;$('#finderBack').onclick=beginChange;
$('#importBtn').onclick=()=>$('#fileInput').click();$('#fileInput').onchange=async e=>{let f=e.target.files?.[0];if(!f)return;try{let slot=S.selected==='GROOVE'?(S.sounds.B?'A':'B'):S.selected,o=await importSound(f),sounds=copySoundRefs(S.sounds);sounds[slot]=o;let c=copy(S.groove);if(!c.events.some(x=>x.slot===slot))c.events.push({slot,beat:3.5,gain:.7,rate:1});S.before=copy(S.groove);S.beforeSounds=copySoundRefs(S.sounds);prepareCompare(c,{label:`SOUND ${slot} → ${f.name}`,sounds})}catch(err){status('IMPORT',err.message)}};
['shapeStart','shapeEnd','shapeRate','shapeGain'].forEach(id=>$('#'+id).oninput=syncShape);$('#reverseBtn').onclick=()=>{let o=S.shapeDraft;if(!o)return;o.shape.reverse=!o.shape.reverse;$('#reverseBtn').textContent=o.shape.reverse?'REVERSED':'REVERSE';drawShape()};$('#shapeHear').onclick=hearShape;$('#shapeCompare').onclick=compareShape;$('#shapeCancel').onclick=cancelShape;
$('#historyBtn').onclick=()=>{renderHistory();setView('history','history')};$('#historyBack').onclick=()=>setView('listen','listen');$('#crumbGroove').onclick=()=>{renderHistory();setView('history','history')};
$('#crumbObject').onclick=()=>setView('listen','listen');$('#aiBtn').onclick=()=>openSheet('aiSheet');$('#connectBtn').onclick=connectAI;
timingPointer();

async function init(){
  await initAudio();S.groove=defaultGroove();renderMain();renderHistory();loadCatalog();status('READY','tap LISTEN first');
}
init();
})();
