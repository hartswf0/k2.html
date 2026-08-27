const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const hash=s=>{let h=2166136261;for(let c of s){h^=c.charCodeAt(0);h=Math.imul(h,16777619)}return h>>>0};
const rng=seed=>{let x=seed||1;return()=>((x^=x<<13,x^=x>>>17,x^=x<<5)>>>0)/4294967296};
const copy=o=>JSON.parse(JSON.stringify(o));
const EARS="https://earsketch.gatech.edu/backend-static/";
const HOUND_TEXT={
  kin:"Find relatives. Preserve timbre or lineage. Do not add density.",
  structure:"Ignore names. Find contrasting material with a useful attack, decay, or envelope.",
  lineage:"Search the current groove language and nearby musical lineages.",
  alien:"Move far away. Prefer material that creates a new relation rather than a better match.",
  morph:"Keep the source. Propose what this sound itself could become after cutting, stretching, reversing, or filtering."
};
const S={
  ctx:null,master:null,playing:false,nodes:[],catalog:[],catalogReady:false,
  slots:{A:null,B:null},activeSlot:"A",hound:"kin",children:[],selectedChild:-1,sediment:[],
  groove:null,generation:1,rendered:null,analysis:null,
  ai:{key:"",model:"",ready:false,tx:null,tab:"result"}
};

function status(k,t,busy=false){$("#statusKey").textContent=k;$("#statusText").textContent=t;$("#pulse").classList.toggle("busy",busy);$("#progress").classList.toggle("busy",busy)}
function open(id){$("#"+id).classList.add("open")}function close(id){$("#"+id).classList.remove("open")}
$$('[data-close]').forEach(b=>b.onclick=()=>close(b.dataset.close));

async function ensureCtx(){
  if(!S.ctx)S.ctx=new (window.AudioContext||window.webkitAudioContext)();
  await S.ctx.resume();
  if(!S.master){S.master=S.ctx.createGain();S.master.gain.value=.82;S.master.connect(S.ctx.destination)}
  return S.ctx;
}
function makeSeedBuffer(kind){
  let c=S.ctx,sr=c.sampleRate,len=Math.floor(sr*(kind==="kick"?.32:.48)),b=c.createBuffer(1,len,sr),d=b.getChannelData(0);
  if(kind==="kick"){
    let ph=0;
    for(let i=0;i<len;i++){let t=i/sr,f=120*Math.pow(42/120,t/.22);ph+=2*Math.PI*f/sr;d[i]=Math.sin(ph)*Math.exp(-t*14)}
  }else{
    let r=rng(991);
    for(let i=0;i<len;i++){let t=i/sr,n=(r()*2-1)*Math.exp(-t*9),tone=Math.sin(2*Math.PI*1200*t)*Math.exp(-t*12);d[i]=.55*n+.45*tone}
  }
  return b;
}
async function initSeeds(){
  await ensureCtx();
  S.slots.A={id:"seed-kick",name:"SEED KICK",state:"READY",kind:"local",buffer:makeSeedBuffer("kick"),shape:{start:0,end:1,rate:1,gain:1,reverse:false},trace:["LOCAL SEED"]};
  updateSlots();
}
async function loadCatalog(){
  try{
    let r=await fetch(EARS+"audio-standard_4.json");if(!r.ok)throw Error(r.status);
    S.catalog=(await r.json()).filter(x=>x&&x.name&&x.path);S.catalogReady=true;
    $("#catalogCount").textContent=`ES ${S.catalog.length}`;status("READY",`${S.catalog.length} sounds available`);
  }catch(e){$("#catalogCount").textContent="ES OFF";status("CATALOG","EarSketch unavailable")}
}
async function loadSound(meta){
  await ensureCtx();status("LOAD",meta.name,true);
  let r=await fetch(EARS+meta.path);if(!r.ok)throw Error(`sample ${r.status}`);
  let a=await r.arrayBuffer(),b=await S.ctx.decodeAudioData(a);
  return {id:meta.name,name:meta.name,state:"READY",kind:"earsketch",meta,buffer:b,shape:{start:0,end:1,rate:1,gain:1,reverse:false},trace:[`FOUND ${meta.name}`,"FETCHED","DECODED"]};
}
async function importFile(file){
  await ensureCtx();let a=await file.arrayBuffer(),b=await S.ctx.decodeAudioData(a);
  return{id:`local-${Date.now()}`,name:file.name,state:"READY",kind:"import",buffer:b,shape:{start:0,end:1,rate:1,gain:1,reverse:false},trace:["IMPORTED",file.name]};
}
function updateSlots(){
  for(let k of ["A","B"]){let o=S.slots[k];$(`#slot${k}Name`).textContent=o?o.name:"EMPTY";$(`#slot${k}State`).textContent=o?`${o.kind.toUpperCase()} · ${o.state}`:"ADD ONLY IF USEFUL"}
  $$(".soundSlot").forEach(b=>b.classList.toggle("active",b.dataset.slot===S.activeSlot));
}

function defaultGroove(){
  return{id:`g-${Date.now()}`,name:"GROOVE 01",bars:2,bpm:118,beats:8,events:[
    {slot:"A",beat:0,gain:1,rate:1},{slot:"A",beat:2.75,gain:.86,rate:1},{slot:"A",beat:4,gain:.92,rate:1},{slot:"A",beat:6.5,gain:.78,rate:1},{slot:"A",beat:7.25,gain:.62,rate:1}
  ],parents:[],mutation:"seed"};
}
function sortEvents(g){g.events.sort((a,b)=>a.beat-b.beat);return g}
function normalizeGroove(g){g.events=g.events.filter(e=>S.slots[e.slot]&&e.beat>=0&&e.beat<g.beats);return sortEvents(g)}
function gapStats(g){
  let e=[...g.events].sort((a,b)=>a.beat-b.beat),gaps=[];
  if(!e.length)return{max:g.beats,mean:g.beats};
  for(let i=0;i<e.length;i++){let a=e[i].beat,b=i===e.length-1?g.beats:e[i+1].beat;gaps.push(b-a)}
  return{max:Math.max(...gaps),mean:gaps.reduce((a,b)=>a+b,0)/gaps.length};
}
function patternMetrics(g){
  let gap=gapStats(g),off=g.events.filter(e=>Math.abs(e.beat*2-Math.round(e.beat*2))>.04).length;
  let occupied=new Set(g.events.map(e=>Math.floor(e.beat*4))).size,total=g.beats*4;
  return{events:g.events.length,silence:Math.max(0,1-occupied/total),maxGap:gap.max,offgrid:g.events.length?off/g.events.length:0};
}

function sampleAt(obj,frac){
  let b=obj.buffer,d=b.getChannelData(0),shape=obj.shape||{start:0,end:1,rate:1,gain:1,reverse:false};
  let a=Math.floor(shape.start*d.length),z=Math.max(a+1,Math.floor(shape.end*d.length)),span=z-a,x=clamp(frac,0,.999999)*span;
  let i=Math.floor(x),f=x-i,idx=shape.reverse?z-1-i:a+i,idx2=shape.reverse?Math.max(a,idx-1):Math.min(z-1,idx+1);
  return (d[idx]||0)*(1-f)+(d[idx2]||0)*f;
}
function renderPCM(g){
  let sr=22050,dur=g.beats*60/g.bpm+1,n=Math.ceil(dur*sr),out=new Float32Array(n);
  for(let ev of g.events){
    let obj=S.slots[ev.slot];if(!obj?.buffer)continue;
    let sh=obj.shape,srcDur=(sh.end-sh.start)*obj.buffer.duration/(sh.rate*(ev.rate||1)),start=Math.floor(ev.beat*60/g.bpm*sr),count=Math.min(n-start,Math.floor(srcDur*sr)),gain=(sh.gain||1)*(ev.gain||1);
    for(let i=0;i<count;i++){let frac=i/Math.max(1,count);out[start+i]+=sampleAt(obj,frac)*gain}
  }
  let peak=0;for(let v of out)peak=Math.max(peak,Math.abs(v));if(peak>1){let k=.98/peak;for(let i=0;i<out.length;i++)out[i]*=k}
  return{pcm:out,sr,dur:g.beats*60/g.bpm};
}
function analyze(render,g){
  let {pcm,sr}=render,win=512,hop=256,rms=[],prev=0,total=0,bright=0;
  for(let i=0;i<pcm.length;i++){let v=pcm[i];total+=v*v;let d=v-prev;bright+=d*d;prev=v}
  for(let i=0;i+win<pcm.length;i+=hop){let s=0;for(let j=0;j<win;j++)s+=pcm[i+j]*pcm[i+j];rms.push(Math.sqrt(s/win))}
  let max=Math.max(.000001,...rms),thr=max*.075,silent=rms.filter(x=>x<thr).length/Math.max(1,rms.length),onsets=0,last=0;
  for(let i=1;i<rms.length;i++){let rise=rms[i]-rms[i-1];if(rise>max*.16&&i-last>2){onsets++;last=i}}
  let p=patternMetrics(g),r=Math.sqrt(total/Math.max(1,pcm.length)),brightness=Math.sqrt(bright/Math.max(1,total));
  return{rms:r,silence:Math.max(p.silence,silent),onsets:Math.max(onsets,g.events.length),brightness,offgrid:p.offgrid,maxGap:p.maxGap};
}
function metric(k,v){return`<div class="metric"><small>${k}</small><b>${v}</b></div>`}
function updateAnalysis(){
  S.rendered=renderPCM(S.groove);S.analysis=analyze(S.rendered,S.groove);
  let a=S.analysis;
  $("#metrics").innerHTML=metric("EVENTS",S.groove.events.length)+metric("SILENCE",`${Math.round(a.silence*100)}%`)+metric("MAX GAP",`${a.maxGap.toFixed(2)}b`)+metric("CROOKED",`${Math.round(a.offgrid*100)}%`);
  $("#grooveMeta").textContent=`${S.groove.bars} bars · ${S.groove.events.length} events · ${Math.round(a.silence*100)}% silence`;
  drawGroove();drawSpectrum();
}
function canvasFit(c,h){let dpr=devicePixelRatio||1,w=c.clientWidth;c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);let x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);return{x,w,h}}
function drawGroove(){
  let c=$("#grooveCanvas"),{x,w,h}=canvasFit(c,170);x.clearRect(0,0,w,h);x.fillStyle="#fff";x.fillRect(0,0,w,h);
  x.strokeStyle="rgba(0,0,0,.16)";x.lineWidth=1;for(let b=0;b<=S.groove.beats;b++){let px=b/S.groove.beats*w;x.beginPath();x.moveTo(px,0);x.lineTo(px,h);x.stroke()}
  let pcm=S.rendered?.pcm;if(pcm){x.strokeStyle="#000";x.lineWidth=1.2;x.beginPath();for(let px=0;px<w;px++){let i=Math.floor(px/w*pcm.length),y=h*.56-pcm[i]*h*.26;if(px===0)x.moveTo(px,y);else x.lineTo(px,y)}x.stroke()}
  for(let i=0;i<S.groove.events.length;i++){let e=S.groove.events[i],px=e.beat/S.groove.beats*w,y=e.slot==="A"?36:132;x.fillStyle=e.slot==="A"?"#000":"#efff00";x.strokeStyle="#000";x.lineWidth=2;x.beginPath();x.arc(px,y,11,0,Math.PI*2);x.fill();x.stroke()}
}
function drawSpectrum(){
  let c=$("#spectrumCanvas"),{x,w,h}=canvasFit(c,58),pcm=S.rendered?.pcm;if(!pcm)return;x.fillStyle="#fff";x.fillRect(0,0,w,h);
  let cols=Math.min(72,Math.floor(w/4)),bins=18,N=128;
  for(let cx=0;cx<cols;cx++){let center=Math.floor(cx/cols*pcm.length),vals=[];for(let k=0;k<bins;k++){let re=0,im=0;for(let n=0;n<N;n++){let idx=center+n-N/2,v=idx>=0&&idx<pcm.length?pcm[idx]:0,ang=2*Math.PI*k*n/N;re+=v*Math.cos(ang);im-=v*Math.sin(ang)}vals.push(Math.sqrt(re*re+im*im))}
    let mx=Math.max(.00001,...vals);for(let k=0;k<bins;k++){let q=Math.pow(vals[k]/mx,.5),gray=Math.round(255*(1-q));x.fillStyle=`rgb(${gray},${gray},${gray})`;x.fillRect(cx*w/cols,h-(k+1)*h/bins,w/cols+1,h/bins+1)}
  }
}

function scheduleObject(obj,when,ev){
  let b=obj.buffer,src=S.ctx.createBufferSource(),g=S.ctx.createGain(),sh=obj.shape,offset=sh.start*b.duration,end=sh.end*b.duration,dur=Math.max(.01,end-offset),rate=sh.rate*(ev.rate||1);
  if(sh.reverse){let rb=reverseBuffer(obj);src.buffer=rb;offset=(1-sh.end)*b.duration}else src.buffer=b;
  src.playbackRate.value=rate;g.gain.value=(sh.gain||1)*(ev.gain||1);src.connect(g).connect(S.master);src.start(when,offset,Math.min(dur,src.buffer.duration-offset));S.nodes.push(src,g)
}
function reverseBuffer(obj){
  if(obj.reverseBuffer)return obj.reverseBuffer;let b=obj.buffer,r=S.ctx.createBuffer(b.numberOfChannels,b.length,b.sampleRate);
  for(let ch=0;ch<b.numberOfChannels;ch++){let a=b.getChannelData(ch),d=r.getChannelData(ch);for(let i=0;i<a.length;i++)d[i]=a[a.length-1-i]}
  obj.reverseBuffer=r;return r;
}
async function playGroove(g=S.groove){
  if(S.playing){stop();return}await ensureCtx();S.playing=true;$("#playBtn").textContent="■ STOP";status("PLAY",`${g.name||g.mutation}`,true);let start=S.ctx.currentTime+.05,spb=60/g.bpm;
  for(let ev of g.events){let obj=S.slots[ev.slot];if(obj?.buffer)scheduleObject(obj,start+ev.beat*spb,ev)}
  let end=start+g.beats*spb;let timer=setInterval(()=>{if(!S.playing){clearInterval(timer);return}if(S.ctx.currentTime>=end){clearInterval(timer);stop()}},80)
}
function stop(){if(!S.playing)return;S.playing=false;S.nodes.forEach(n=>{try{n.disconnect?.()}catch{}});S.nodes=[];$("#playBtn").textContent="▶ LISTEN";status("READY","listen → compare → keep")}

function mutate(g,type,seed){
  let r=rng(seed||Date.now()),n=copy(g);n.id=`g-${Date.now()}-${Math.floor(r()*9999)}`;n.parents=[g.id];n.mutation=type;
  let e=n.events;
  if(type==="SPARSE"){e=e.filter((_,i)=>i===0||r()>.38);if(e.length>5)e=e.slice(0,5)}
  if(type==="CROOKED"){e.forEach((x,i)=>{if(i&&r()>.35)x.beat=clamp(x.beat+(r()>.5?.125:-.125),0,n.beats-.05)})}
  if(type==="BREATH"){e.sort((a,b)=>a.beat-b.beat);let out=[];for(let x of e){if(!out.length||x.beat-out.at(-1).beat>=.75||r()>.75)out.push(x)}e=out}
  if(type==="ECHO"&&e.length){let x=copy(e[Math.floor(r()*e.length)]);x.beat=clamp(x.beat+.75,0,n.beats-.1);x.gain*=.45;e.push(x)}
  if(type==="CALL"){if(S.slots.B){e=e.map((x,i)=>({...x,slot:i%2?"B":"A"}));if(e.length<4)e.push({slot:"B",beat:3.5,gain:.72,rate:1})}else type="BREATH"}
  if(type==="ALIEN"){e.forEach(x=>{if(r()>.5)x.rate=pickR(r,[.5,.75,1,1.5,2]);if(r()>.72)x.beat=clamp(Math.round(x.beat*4)/4+(r()-.5)*.5,0,n.beats-.05)});if(e.length>6)e=e.filter(()=>r()>.28)}
  if(type==="VOID"){e=e.filter((x,i)=>i%2===0);if(e.length>3)e=e.slice(0,3)}
  n.events=e;return normalizeGroove(n)
}
function pickR(r,a){return a[Math.floor(r()*a.length)]}
function makeChildren(){
  let base=S.groove,types=S.slots.B?["SPARSE","CROOKED","CALL","ALIEN"]:["SPARSE","BREATH","CROOKED","VOID"];
  S.children=types.map((t,i)=>{let g=mutate(base,t,hash(base.id+t+S.generation+i)),a=analyze(renderPCM(g),g);return{g,a}});
  S.selectedChild=-1;S.generation++;renderChildren();status("CHILDREN","four different directions")}
function renderChildren(){
  let h=$("#children");h.innerHTML="";S.children.forEach((c,i)=>{let b=document.createElement("button");b.className="child"+(S.selectedChild===i?" sel":"");b.innerHTML=`<b>${c.g.mutation}</b><small>${c.g.events.length} events · ${Math.round(c.a.silence*100)}% silence<br>${Math.round(c.a.offgrid*100)}% off-grid · gap ${c.a.maxGap.toFixed(2)}b</small><div class="miniPattern"></div>`;let p=b.querySelector(".miniPattern");c.g.events.forEach(e=>{let m=document.createElement("i");m.className="miniHit";m.style.left=`${e.beat/c.g.beats*100}%`;m.style.height=e.slot==="A"?"14px":"8px";p.append(m)});b.onclick=()=>{S.selectedChild=i;renderChildren();$("#keepBtn").disabled=$("#breedBtn").disabled=$("#killBtn").disabled=false;playGroove(c.g)};h.append(b)})
}
function keepChild(){
  if(S.selectedChild<0)return;let child=S.children[S.selectedChild].g,parent=S.groove;S.sediment.unshift({g:copy(parent),analysis:S.analysis,time:Date.now()});S.groove=copy(child);S.groove.name=`GROOVE ${String(S.sediment.length+1).padStart(2,"0")}`;S.children=[];S.selectedChild=-1;renderChildren();renderSediment();updateAnalysis();$("#keepBtn").disabled=$("#breedBtn").disabled=$("#killBtn").disabled=true;status("KEPT",`${S.groove.mutation} became current`)}
function breedChild(){
  if(S.selectedChild<0)return;let a=S.groove,b=S.children[S.selectedChild].g,n=copy(a),seen=new Set();n.events=[];[...a.events,...b.events].sort((x,y)=>x.beat-y.beat).forEach((e,i)=>{let key=Math.round(e.beat*8)+"-"+e.slot;if(!seen.has(key)&&(i%2===0||n.events.length<3)){seen.add(key);n.events.push(copy(e))}});while(n.events.length>6)n.events.splice(1+Math.floor(Math.random()*(n.events.length-1)),1);n.id=`breed-${Date.now()}`;n.parents=[a.id,b.id];n.mutation="BREED";S.sediment.unshift({g:copy(a),analysis:S.analysis,time:Date.now()});S.groove=normalizeGroove(n);S.children=[];S.selectedChild=-1;renderChildren();renderSediment();updateAnalysis();status("BREED","parent relationships recombined")}
function killChild(){if(S.selectedChild<0)return;let old=S.children[S.selectedChild].g.mutation,types=["SPARSE","BREATH","CROOKED","ECHO","ALIEN","VOID","CALL"],t=types[Math.floor(Math.random()*types.length)],g=mutate(S.groove,t,Date.now()),a=analyze(renderPCM(g),g);S.children[S.selectedChild]={g,a};S.selectedChild=-1;renderChildren();$("#keepBtn").disabled=$("#breedBtn").disabled=$("#killBtn").disabled=true;status("KILL",`${old} replaced by ${t}`)}
function renderSediment(){
  $("#sedimentCount").textContent=`${S.sediment.length} kept`;let h=$("#sedimentStrip");h.innerHTML="";S.sediment.forEach((s,i)=>{let b=document.createElement("button");b.className="stone";b.innerHTML=`<b>${s.g.name||"GROOVE"}</b><small>${s.g.mutation}<br>${s.g.events.length} events · ${Math.round((s.analysis?.silence||0)*100)}% silence</small>`;b.onclick=()=>{S.sediment.unshift({g:copy(S.groove),analysis:S.analysis,time:Date.now()});S.groove=copy(s.g);updateAnalysis();renderSediment();status("EXCAVATE",S.groove.name)};h.append(b)})
}

function localChat(text){
  let q=text.toLowerCase(),type=/empty|silence|space|less|breathe|breath/.test(q)?"BREATH":/crooked|late|early|swing|drag|push/.test(q)?"CROOKED":/alien|strange|unrecognizable|destroy/.test(q)?"ALIEN":/echo|answer|reply/.test(q)?(S.slots.B?"CALL":"ECHO"):/remove|sparse|thin/.test(q)?"SPARSE":"ECHO";
  let child=mutate(S.groove,type,hash(text+Date.now()));S.children=[{g:child,a:analyze(renderPCM(child),child)},...S.children.slice(0,3)];S.selectedChild=0;renderChildren();$("#keepBtn").disabled=$("#breedBtn").disabled=$("#killBtn").disabled=false;playGroove(child);status("CHAT",`${type} · rendered and measured`)
}
async function api(path,opt={}){let r=await fetch("https://api.openai.com/v1"+path,{...opt,headers:{Authorization:`Bearer ${S.ai.key}`,"Content-Type":"application/json"}}),txt=await r.text(),d;try{d=JSON.parse(txt)}catch{d={raw:txt}}if(!r.ok)throw Object.assign(Error(d?.error?.message||`HTTP ${r.status}`),{data:d});return d}
function output(d){if(typeof d?.output_text==="string")return d.output_text;let a=[];for(let i of d?.output||[])for(let c of i.content||[])if(typeof c?.text==="string")a.push(c.text);return a.join("\n")}
async function connectAI(){
  S.ai.key=$("#apiKey").value.trim();if(!S.ai.key)return;status("OPENAI","checking model",true);
  try{let d=await api("/models"),ids=(d.data||[]).map(x=>x.id).filter(x=>/^gpt-5/.test(x)&&!/(audio|realtime|image|codex|transcribe)/i.test(x));S.ai.model=ids.find(x=>x.includes("5.6-sol"))||ids[0];if(!S.ai.model)throw Error("no compatible model");let p=await api("/responses",{method:"POST",body:JSON.stringify({model:S.ai.model,input:"Reply OK",max_output_tokens:64,store:false})});if(!/OK/i.test(output(p)))throw Error("ping failed");S.ai.ready=true;$("#aiBtn").classList.add("ready");$("#aiBtn").textContent="AI·ON";$("#apiMsg").textContent=`READY · ${S.ai.model}`;status("READY","AI observer available")}catch(e){$("#apiMsg").textContent=e.message;status("AI ERROR",e.message)}
}
async function aiChat(text){
  let req={model:S.ai.model,input:`You are a groove observer. Transform a sparse 2-bar groove. Never increase events above 7. Preserve or increase negative space unless explicitly asked otherwise.
Instruction: ${text}
Current groove: ${JSON.stringify(S.groove)}
Measured render: ${JSON.stringify(S.analysis)}
Available sound B: ${!!S.slots.B}
Return one mutation recipe.`,max_output_tokens:900,store:false,text:{format:{type:"json_schema",name:"groove_mutation",strict:true,schema:{type:"object",properties:{name:{type:"string"},events:{type:"array",maxItems:7,items:{type:"object",properties:{slot:{type:"string",enum:["A","B"]},beat:{type:"number",minimum:0,maximum:7.99},gain:{type:"number",minimum:.05,maximum:1.5},rate:{type:"number",minimum:.25,maximum:3}},required:["slot","beat","gain","rate"],additionalProperties:false}},note:{type:"string"}},required:["name","events","note"],additionalProperties:false}}}};
  status("OPENAI","mutate → render → measure",true);
  try{let resp=await api("/responses",{method:"POST",body:JSON.stringify(req)}),obj=JSON.parse(output(resp)),g=copy(S.groove);g.id=`ai-${Date.now()}`;g.mutation=obj.name;g.events=obj.events.filter(e=>S.slots[e.slot]);normalizeGroove(g);let a=analyze(renderPCM(g),g);S.children=[{g,a},...S.children.slice(0,3)];S.selectedChild=0;S.ai.tx={result:{recipe:obj,measured:a},response:resp,request:req};renderChildren();renderTx();$("#keepBtn").disabled=$("#breedBtn").disabled=$("#killBtn").disabled=false;playGroove(g);status("AI CHILD",`${obj.note} · ${Math.round(a.silence*100)}% silence`)}catch(e){status("AI→LOCAL",e.message);localChat(text)}
}
function renderTx(){if(!S.ai.tx)return;let v=S.ai.tx[S.ai.tab];$("#txBody").textContent=typeof v==="string"?v:JSON.stringify(v,null,2);$$('[data-tx]').forEach(b=>b.classList.toggle("on",b.dataset.tx===S.ai.tab))}
function doChat(){let t=$("#chatInput").value.trim();if(!t)return;S.ai.ready?aiChat(t):localChat(t)}

function forageScore(meta,q,hound){
  let text=[meta.name,meta.folder,meta.genre,meta.instrument,meta.artist].filter(Boolean).join(" ").toLowerCase(),words=q.toLowerCase().split(/[^a-z0-9]+/).filter(x=>x.length>2),s=0;
  for(let w of words)if(text.includes(w))s+=5;
  let active=S.slots[S.activeSlot],abase=active?.name?.toLowerCase()||"";
  if(hound==="kin"){for(let w of abase.split(/[_\s]+/))if(w.length>3&&text.includes(w))s+=4}
  if(hound==="structure"){if(/drum|perc|hit|kick|snare|hat|clap|shaker|pluck/.test(text))s+=5}
  if(hound==="lineage"){for(let w of (S.groove?.mutation||"").toLowerCase().split(/\s+/))if(text.includes(w))s+=2}
  if(hound==="alien")s+=hash(meta.name)%19/3;
  return s;
}
function renderCandidates(){
  let q=$("#soundSearch").value.trim(),list=S.catalog.map(m=>[forageScore(m,q,S.hound),m]).filter(x=>q?x[0]>0:true).sort((a,b)=>b[0]-a[0]);
  if(S.hound==="alien")list.sort((a,b)=>(hash(b[1].name+S.activeSlot)%1000)-(hash(a[1].name+S.activeSlot)%1000));
  list=list.slice(0,80);$("#candidateCount").textContent=S.catalogReady?`${list.length} candidates · ${S.hound.toUpperCase()} HOUND`:"catalog loading…";let h=$("#candidateList");h.innerHTML="";
  if(S.hound==="morph"){let ops=["KEEP ATTACK / STRETCH BODY","REVERSE TAIL","HALF RATE / TRIM","DOUBLE RATE / GHOST","CUT TO FIRST 18%","REVERSE + 0.75×"];for(let op of ops){let b=document.createElement("button");b.className="candidate";b.innerHTML=`<b>MORPH</b><small>${op}</small>`;b.onclick=()=>applyMorph(op);h.append(b)}return}
  for(let [,m] of list){let b=document.createElement("button");b.className="candidate";b.innerHTML=`<b>${m.name}</b><small>${m.folder||m.instrument||m.genre||"EarSketch"}</small>`;b.onclick=async()=>{try{let o=await loadSound(m);S.slots[S.activeSlot]=o;updateSlots();normalizeGroove(S.groove);if(!S.groove.events.some(e=>e.slot===S.activeSlot))S.groove.events.push({slot:S.activeSlot,beat:S.activeSlot==="A"?0:3.5,gain:.75,rate:1});updateAnalysis();close("forageSheet");status("READY",`${S.activeSlot} · ${m.name}`)}catch(e){status("FAILED",e.message)}};h.append(b)}
}
function applyMorph(op){
  let o=S.slots[S.activeSlot];if(!o)return;
  if(/FIRST 18/.test(op)){o.shape.start=0;o.shape.end=.18}
  if(/HALF RATE/.test(op))o.shape.rate=.5;if(/DOUBLE RATE/.test(op))o.shape.rate=2;if(/0.75/.test(op))o.shape.rate=.75;
  if(/REVERSE/.test(op))o.shape.reverse=!o.shape.reverse;if(/STRETCH BODY/.test(op)){o.shape.end=.55;o.shape.rate=.5}
  o.trace.push(`MORPH ${op}`);updateAnalysis();close("forageSheet");openSculpt()
}
function openForage(){$("#forageTarget").textContent=S.activeSlot;$("#houndPrompt").textContent=HOUND_TEXT[S.hound];open("forageSheet");renderCandidates()}
function openSculpt(){
  let o=S.slots[S.activeSlot];if(!o)return;$("#sculptTarget").textContent=`${S.activeSlot} · ${o.name}`;$("#shapeStart").value=Math.round(o.shape.start*100);$("#shapeEnd").value=Math.round(o.shape.end*100);$("#shapeRate").value=Math.round(o.shape.rate*100);$("#shapeGain").value=Math.round(o.shape.gain*100);$("#reverseBtn").textContent=o.shape.reverse?"REVERSED":"REVERSE";$("#shapeTrace").textContent=o.trace.join(" → ");drawSample();open("sculptSheet")
}
function drawSample(){let o=S.slots[S.activeSlot],c=$("#sampleCanvas"),{x,w,h}=canvasFit(c,180);x.fillStyle="#fff";x.fillRect(0,0,w,h);if(!o?.buffer)return;let d=o.buffer.getChannelData(0);x.strokeStyle="#000";x.lineWidth=1;x.beginPath();for(let px=0;px<w;px++){let i=Math.floor(px/w*d.length),y=h/2-d[i]*h*.42;if(px===0)x.moveTo(px,y);else x.lineTo(px,y)}x.stroke();x.fillStyle="rgba(239,255,0,.55)";x.fillRect(o.shape.start*w,0,(o.shape.end-o.shape.start)*w,h);x.strokeStyle="#000";x.strokeRect(o.shape.start*w,0,(o.shape.end-o.shape.start)*w,h)}
function syncShape(){
  let o=S.slots[S.activeSlot];if(!o)return;o.shape.start=+$("#shapeStart").value/100;o.shape.end=+$("#shapeEnd").value/100;if(o.shape.end<=o.shape.start+.03)o.shape.end=Math.min(1,o.shape.start+.03);o.shape.rate=+$("#shapeRate").value/100;o.shape.gain=+$("#shapeGain").value/100;o.trace.push(`SHAPE ${Math.round(o.shape.start*100)}–${Math.round(o.shape.end*100)}% · ${o.shape.rate.toFixed(2)}×`);drawSample();updateAnalysis();$("#shapeTrace").textContent=o.trace.slice(-5).join(" → ")
}
async function audition(){let o=S.slots[S.activeSlot];if(!o)return;await ensureCtx();let ev={rate:1,gain:1};scheduleObject(o,S.ctx.currentTime+.03,ev)}
function bounce(){
  let o=S.slots[S.activeSlot];
  if(!o)return;
  let sh=o.shape,
      sr=o.buffer.sampleRate,
      start=Math.floor(sh.start*o.buffer.length),
      end=Math.floor(sh.end*o.buffer.length),
      len=Math.max(1,Math.floor((end-start)/sh.rate)),
      b=S.ctx.createBuffer(o.buffer.numberOfChannels,len,sr);
  for(let ch=0;ch<o.buffer.numberOfChannels;ch++){
    let src=o.buffer.getChannelData(ch),dst=b.getChannelData(ch);
    for(let i=0;i<len;i++){
      let pos=start+i*sh.rate,ix=Math.floor(pos),v=src[Math.min(end-1,ix)]||0;
      dst[sh.reverse?len-1-i:i]=v*sh.gain;
    }
  }
  S.slots[S.activeSlot]={id:`bounce-${Date.now()}`,name:`${o.name} · BOUNCE`,state:"READY",kind:"bounce",buffer:b,shape:{start:0,end:1,rate:1,gain:1,reverse:false},trace:[...o.trace,"BOUNCED LOCAL"]};
  updateSlots();updateAnalysis();close("sculptSheet");status("BOUNCED",`${S.activeSlot} is now local`);
}

$$(".soundSlot").forEach(b=>{b.onclick=()=>{S.activeSlot=b.dataset.slot;updateSlots();let o=S.slots[S.activeSlot];if(o)openSculpt();else openForage()}});
$$('[data-hound]').forEach(b=>b.onclick=()=>{S.hound=b.dataset.hound;$$('[data-hound]').forEach(x=>x.classList.toggle("on",x===b));$("#houndPrompt").textContent=HOUND_TEXT[S.hound];renderCandidates()});
$("#soundSearch").oninput=renderCandidates;$("#importBtn").onclick=()=>$("#fileInput").click();$("#fileInput").onchange=async e=>{let f=e.target.files?.[0];if(!f)return;try{S.slots[S.activeSlot]=await importFile(f);updateSlots();if(!S.groove.events.some(x=>x.slot===S.activeSlot))S.groove.events.push({slot:S.activeSlot,beat:3.5,gain:.7,rate:1});updateAnalysis();close("forageSheet")}catch(err){status("IMPORT",err.message)}};
$("#forageBtn").onclick=openForage;$("#playBtn").onclick=()=>playGroove();$("#mutateBtn").onclick=makeChildren;$("#mutateDock").onclick=makeChildren;$("#keepBtn").onclick=keepChild;$("#breedBtn").onclick=breedChild;$("#killBtn").onclick=killChild;
$("#chatGo").onclick=doChat;$("#chatInput").onkeydown=e=>{if(e.key==="Enter"){e.preventDefault();doChat()}};
$("#lessBtn").onclick=()=>{if(S.groove.events.length>1){let idx=1+Math.floor(Math.random()*(S.groove.events.length-1));S.groove.events.splice(idx,1);updateAnalysis();status("LESS",`${S.groove.events.length} events remain`)}};
$("#moreBtn").onclick=()=>{let slot=S.slots.B&&Math.random()>.5?"B":"A",beats=[1.25,2.25,3.5,5.25,6.75],beat=beats.find(b=>!S.groove.events.some(e=>Math.abs(e.beat-b)<.2));if(beat!=null&&S.groove.events.length<7){S.groove.events.push({slot,beat,gain:.6,rate:1});normalizeGroove(S.groove);updateAnalysis()}};
$("#aiBtn").onclick=()=>open("aiSheet");$("#connectBtn").onclick=connectAI;$$('[data-tx]').forEach(b=>b.onclick=()=>{S.ai.tab=b.dataset.tx;renderTx()});
["shapeStart","shapeEnd","shapeRate","shapeGain"].forEach(id=>$("#"+id).oninput=syncShape);$("#reverseBtn").onclick=()=>{let o=S.slots[S.activeSlot];o.shape.reverse=!o.shape.reverse;o.trace.push(o.shape.reverse?"REVERSE":"UNREVERSE");$("#reverseBtn").textContent=o.shape.reverse?"REVERSED":"REVERSE";drawSample();updateAnalysis()};$("#auditionBtn").onclick=audition;$("#bounceBtn").onclick=bounce;

async function init(){
  await initSeeds();S.groove=defaultGroove();updateAnalysis();renderSediment();loadCatalog();
  status("READY","one sound first · mutate after listening")
}
init();
