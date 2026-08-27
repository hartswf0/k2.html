/* ES/TRACE v9 — TUNEHOUNDS / RADIODRAW
   A sample is a first-class object:
   FOUND → FETCHING → DECODED → AUDITIONED → BOUNCED → READY.
   Forage returns several directions; Radiodraw turns waveform gesture into gain shape.
*/

S.lab ||= {
  track:0,
  source:null,
  sourceName:"",
  buffer:null,
  bounced:null,
  state:"EMPTY",
  mode:"select",
  sel:[0,1],
  draw:[],
  rate:1,
  tone:0,
  drive:0,
  space:0,
  grain:0,
  reverse:false,
  audition:null,
  recipe:null,
  forage:[]
};

function labTrack(){let w=S.worlds[S.world];return w?.tracks?.[clamp(S.lab.track||0,0,Math.max(0,(w?.tracks?.length||1)-1))]||null}
function labSetState(next,detail=""){
  S.lab.state=next;
  let el=$("#labState");
  if(el){el.textContent=`${next}${detail?" · "+detail:""}`;el.classList.toggle("sampleStateReady",next==="READY"||next==="BOUNCED");el.classList.toggle("sampleStateFail",next==="FAILED")}
  renderLabManifest();
}
function renderLabManifest(){let el=$("#labManifest");if(!el)return;let t=labTrack(),b=S.lab.buffer||S.lab.bounced,sel=S.lab.sel||[0,1];let bits=[];bits.push(`STATE ${S.lab.state}`);if(b)bits.push(`${b.duration.toFixed(2)}s`);if(b)bits.push(`CUT ${(Math.abs(sel[1]-sel[0])*b.duration).toFixed(2)}s`);bits.push(`RATE ${S.lab.rate.toFixed(2)}×`);if(S.lab.reverse)bits.push("REVERSE");if(Math.abs(S.lab.tone)>.01)bits.push(`TONE ${S.lab.tone>0?"+":""}${S.lab.tone.toFixed(2)}`);if(S.lab.grain>.01)bits.push(`GRAIN ${S.lab.grain.toFixed(2)}`);if(t?.localBuffer)bits.push("LOCAL BOUNCE");el.textContent=bits.join(" · ")}
function labTrace(){let h=$("#labTrace");if(!h)return;let t=labTrack();let src=S.lab.sourceName||t?.sample?.name||t?.label||"none";let ops=[];if(S.lab.sel&&(S.lab.sel[0]>0||S.lab.sel[1]<1))ops.push(`cut ${Math.min(...S.lab.sel).toFixed(2)}→${Math.max(...S.lab.sel).toFixed(2)}`);if(S.lab.reverse)ops.push("reverse");if(S.lab.rate!==1)ops.push(`rate ×${S.lab.rate.toFixed(2)}`);if(Math.abs(S.lab.tone)>.01)ops.push(`tone ${S.lab.tone.toFixed(2)}`);if(S.lab.drive)ops.push(`drive ${S.lab.drive}`);if(S.lab.space)ops.push(`space ${S.lab.space.toFixed(2)}`);if(S.lab.grain)ops.push(`grain ${S.lab.grain.toFixed(2)}`);if(S.lab.draw.length>1)ops.push("Radiodraw envelope");h.innerHTML=`<div class="step"><small>SOURCE</small>${esc(src)}</div><div class="step"><small>OPERATIONS</small>${esc(ops.join(" → ")||"none yet")}</div><div class="step"><small>OUTPUT</small>${t?.localBuffer?`local bounced sound · ${t.localBuffer.duration.toFixed(2)}s`:"not bounced"}</div>`}

function resetLabShape(){S.lab.sel=[0,1];S.lab.draw=[];S.lab.rate=1;S.lab.tone=0;S.lab.drive=0;S.lab.space=0;S.lab.grain=0;S.lab.reverse=false;syncLabControls();drawRadio()}
function syncLabControls(){let pairs=[["rateCtl","rate",v=>`${(+v).toFixed(2)}×`],["toneCtl","tone",v=>`${+v>0?"+":""}${(+v).toFixed(2)}`],["driveCtl","drive",v=>`${v}`],["spaceCtl","space",v=>(+v).toFixed(2)],["grainCtl","grain",v=>(+v).toFixed(2)]];for(let [id,k,fmtv] of pairs){let x=$("#"+id);if(x)x.value=S.lab[k];let out=$("#"+id.replace("Ctl","Val"));if(out)out.textContent=fmtv(S.lab[k])}let r=$("#reverseCtl");if(r){r.classList.toggle("on",S.lab.reverse);r.textContent=`REVERSE · ${S.lab.reverse?"ON":"OFF"}`}renderLabManifest();labTrace()}

async function openSoundLab(trackIndex=null,forageOnly=false){
  let w=S.worlds[S.world];if(!w)return;
  let ti=trackIndex!=null?trackIndex:(S.sel?.track??w.tracks.findIndex(t=>t.sample||t.localBuffer));if(ti<0)ti=S.sel?.track??0;
  S.lab.track=clamp(ti,0,w.tracks.length-1);let t=labTrack();
  $("#labName").textContent=t.localName||t.sample?.name||t.label||"SOUND";$("#labRole").textContent=`${t.role||"track"} · ${t.tag||""}`;
  open("soundLabSheet");resetLabShape();
  if(t.localBuffer){S.lab.buffer=t.localBuffer;S.lab.bounced=t.localBuffer;S.lab.sourceName=t.localName||t.label;labSetState("READY","local bounce");drawRadio()}
  else if(t.sample){await useLabSample(t.sample,false)}
  else{S.lab.buffer=null;S.lab.source=null;S.lab.sourceName="";labSetState("FOUND","no sample yet · forage or import");drawRadio()}
  buildTunehounds(forageOnly?`${t.tag||""} ${t.role||""}`:($("#houndPrompt").value||`${t.tag||""} ${t.role||""}`));
  labTrace();
}

async function useLabSample(sm,assign=true){
  if(!sm)return;S.lab.source=sm;S.lab.sourceName=sm.name;$("#labName").textContent=sm.name;labSetState("FOUND",sm.folder||sm.instrument||"");
  await ensureCtxV83();labSetState("FETCHING",sm.name.slice(0,28));
  let b=await loadESBufV83(sm);if(!b){labSetState("FAILED","sample did not decode");return null}
  S.lab.buffer=b;S.lab.bounced=null;labSetState("DECODED",`${b.duration.toFixed(2)}s`);resetLabShape();drawRadio();
  if(assign){let t=labTrack();t.sample=sm;t.localBuffer=null;t.localName=null;t.drums=false;t.program=t.program??118;t.label=sm.name;t.tag=`EarSketch · ${sm.folder||sm.genre||"sample"}`;render()}
  labSetState("READY","cached");return b
}

function radioXY(ev){let c=$("#radioCanvas"),r=c.getBoundingClientRect();return[clamp((ev.clientX-r.left)/r.width,0,1),clamp((ev.clientY-r.top)/r.height,0,1)]}
function drawRadio(){
  let c=$("#radioCanvas");if(!c)return;let rect=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),W=Math.max(10,Math.floor(rect.width*dpr)),H=Math.max(10,Math.floor(rect.height*dpr));if(c.width!==W||c.height!==H){c.width=W;c.height=H}let g=c.getContext("2d");g.clearRect(0,0,W,H);g.fillStyle="#fff";g.fillRect(0,0,W,H);g.strokeStyle="#000";g.lineWidth=Math.max(1,dpr);
  let b=S.lab.buffer||S.lab.bounced;if(b){let data=b.getChannelData(0),step=Math.max(1,Math.floor(data.length/W)),mid=H/2;g.beginPath();for(let x=0;x<W;x++){let start=x*step,min=1,max=-1;for(let j=0;j<step&&start+j<data.length;j++){let v=data[start+j];if(v<min)min=v;if(v>max)max=v}g.moveTo(x,mid+min*mid*.9);g.lineTo(x,mid+max*mid*.9)}g.stroke()}
  let a=Math.min(...S.lab.sel),z=Math.max(...S.lab.sel);g.fillStyle="rgba(239,255,0,.32)";g.fillRect(a*W,0,(z-a)*W,H);g.strokeStyle="#ff5a36";g.lineWidth=2*dpr;g.beginPath();g.moveTo(a*W,0);g.lineTo(a*W,H);g.moveTo(z*W,0);g.lineTo(z*W,H);g.stroke();
  if(S.lab.draw.length>1){g.strokeStyle="#ff5a36";g.lineWidth=3*dpr;g.beginPath();S.lab.draw.forEach(([x,y],i)=>i?g.lineTo(x*W,y*H):g.moveTo(x*W,y*H));g.stroke()}
  g.fillStyle="#000";g.font=`${8*dpr}px ui-monospace`;g.fillText(S.lab.mode.toUpperCase(),6*dpr,12*dpr)
}
let radioDown=false;
function radioDownFn(ev){radioDown=true;let [x,y]=radioXY(ev);if(S.lab.mode==="draw"){S.lab.draw=[[x,y]]}else S.lab.sel=[x,x];drawRadio();ev.currentTarget.setPointerCapture?.(ev.pointerId)}
function radioMoveFn(ev){if(!radioDown)return;let[x,y]=radioXY(ev);if(S.lab.mode==="draw")S.lab.draw.push([x,y]);else S.lab.sel[1]=x;drawRadio();renderLabManifest()}
function radioUpFn(){radioDown=false;labTrace()}

function rankCatalog(query,exclude=[],limit=6,alien=false){
  if(!S.es.length)return[];let words=String(query).toLowerCase().split(/[^a-z0-9à-ÿ]+/i).filter(x=>x.length>2),ex=exclude.map(x=>String(x).toLowerCase()),r=rng(hash(query+(alien?"alien":"rank")));
  let scored=S.es.map(sm=>{let text=[sm.name,sm.folder,sm.genre,sm.instrument,sm.artist].filter(Boolean).join(" ").toLowerCase(),score=0;for(let w of words)if(text.includes(w))score+=w.length>5?5:2;for(let x of ex)if(x&&text.includes(x))score-=alien?8:2;if(alien)score+=r()*5;return[score+r()*.25,sm]}).sort((a,b)=>b[0]-a[0]);
  let positive=scored.filter(x=>alien?x[0]>0:x[0]>0.2);if(!positive.length)positive=scored;return positive.slice(0,limit).map(x=>x[1])
}
function currentForageText(extra=""){let w=S.worlds[S.world],t=labTrack(),sm=S.lab.source;return [extra,w?.prompt,(w?.tags||[]).join(" "),t?.role,t?.tag,sm?.name,sm?.folder,sm?.instrument,sm?.genre].filter(Boolean).join(" ")}
function buildTunehounds(extra="",aiPlan=null){
  let t=labTrack(),base=currentForageText(extra),currentWords=[S.lab.source?.name,S.lab.source?.folder,t?.label,t?.tag].filter(Boolean);
  let kinQ=aiPlan?.kin?.join(" ")||base,structureQ=aiPlan?.structure?.join(" ")||`${t?.role||"rhythm"} transient percussion texture attack pulse`,lineageQ=aiPlan?.lineage?.join(" ")||`${S.worlds[S.world]?.tags?.slice(0,5).join(" ")} ${extra}`,alienQ=aiPlan?.alien?.join(" ")||`${distant(base,5).join(" ")} found sound field noise`; 
  let groups=[
    ["KIN HOUND","similar material",rankCatalog(kinQ,currentWords,6,false)],
    ["STRUCTURE HOUND","similar attack / role",rankCatalog(structureQ,currentWords,6,false)],
    ["LINEAGE HOUND","active musical lineage",rankCatalog(lineageQ,currentWords,6,false)],
    ["ALIEN HOUND","deliberately distant",rankCatalog(alienQ,currentWords,6,true)]
  ];S.lab.forage=groups;renderHounds(groups)
}
function renderHounds(groups){let h=$("#houndTray");if(!h)return;h.innerHTML="";for(let [title,why,list] of groups){let g=document.createElement("div");g.className="houndGroup";g.innerHTML=`<div class="houndTitle"><b>${esc(title)}</b><span>${esc(why)}</span></div><div class="houndList"></div>`;let l=g.querySelector(".houndList");if(!list.length){l.innerHTML=`<button class="houndSound" disabled><b>CATALOG LOADING</b><small>EarSketch index has not arrived yet.</small></button>`}for(let sm of list){let b=document.createElement("button");b.className="houndSound";b.innerHTML=`<b>${esc(sm.name)}</b><small>${esc(sm.instrument||sm.genre||sm.folder||"")}</small>`;b.onclick=async()=>{b.classList.add("loading");b.querySelector("small").textContent="FETCHING → DECODE";let ok=await useLabSample(sm,true);b.classList.remove("loading");if(ok){b.querySelector("small").textContent="READY · now sculpt it"}};l.append(b)}g.append(l);h.append(g)}
  let morph=document.createElement("div");morph.className="houndGroup";morph.innerHTML=`<div class="houndTitle"><b>MORPH HOUND</b><span>what this sound could become</span></div><div class="morphList"><button data-morph="attack">KEEP ATTACK</button><button data-morph="freeze">FREEZE BODY</button><button data-morph="grain">GRAIN</button><button data-morph="dark">DARK</button><button data-morph="brittle">BRITTLE</button><button data-morph="reverse">REVERSE</button></div>`;h.append(morph);morph.querySelectorAll("[data-morph]").forEach(b=>b.onclick=()=>applyMorph(b.dataset.morph))
}
function applyMorph(m){let map={attack:{sel:[0,.14],rate:1,tone:.15,drive:2,space:0,grain:0,reverse:false},freeze:{sel:[.12,.8],rate:.28,tone:-.15,drive:1,space:.55,grain:.82,reverse:false},grain:{rate:.7,tone:0,drive:3,space:.15,grain:.78,reverse:false},dark:{rate:.9,tone:-.75,drive:2,space:.25,grain:.1,reverse:false},brittle:{rate:1.15,tone:.72,drive:9,space:.1,grain:.25,reverse:false},reverse:{reverse:!S.lab.reverse}};let r=map[m]||{};Object.assign(S.lab,r);syncLabControls();drawRadio();labTrace()}

function localPromptRecipe(text){let q=text.toLowerCase(),r={};if(/attack|transient/.test(q))r.sel=[0,.16];if(/freeze|frozen|horizon/.test(q)){r.rate=.3;r.grain=.8;r.space=.55}if(/stretch|slow/.test(q))r.rate=.5;if(/fast|short/.test(q))r.rate=1.5;if(/dark|dull|underwater/.test(q))r.tone=-.72;if(/bright|brittle|metal|glass/.test(q))r.tone=.68;if(/destroy|distort|crush/.test(q))r.drive=14;if(/space|distant|lake|echo|reverb/.test(q))r.space=.55;if(/grain|dust|stutter|fragment/.test(q))r.grain=.7;if(/reverse|backward/.test(q))r.reverse=true;return r}
async function sniff(){let text=$("#houndPrompt").value.trim();if(!text)text=currentForageText();Object.assign(S.lab,localPromptRecipe(text));syncLabControls();drawRadio();buildTunehounds(text);if(!S.ready)return;status("OPENAI","Tunehounds are sniffing");let schema={type:"object",properties:{kin:{type:"array",items:{type:"string"},maxItems:4},structure:{type:"array",items:{type:"string"},maxItems:4},lineage:{type:"array",items:{type:"string"},maxItems:4},alien:{type:"array",items:{type:"string"},maxItems:4},shape:{type:"object",properties:{rate:{type:"number",minimum:.25,maximum:2},tone:{type:"number",minimum:-1,maximum:1},drive:{type:"number",minimum:0,maximum:24},space:{type:"number",minimum:0,maximum:1},grain:{type:"number",minimum:0,maximum:1},reverse:{type:"boolean"}},required:["rate","tone","drive","space","grain","reverse"],additionalProperties:false},note:{type:"string"}},required:["kin","structure","lineage","alien","shape","note"],additionalProperties:false};let req={model:S.model,input:`You are Tunehounds. Do NOT invent sample filenames. Return search phrases for several directions through a real audio catalog, plus a sound-shaping recipe. Sound: ${currentForageText(text)}. User: ${text}`,max_output_tokens:600,store:false,text:{format:{type:"json_schema",name:"tunehound_plan",strict:true,schema},verbosity:"low"},reasoning:{effort:"low",summary:"auto"}};try{let resp=await responses(req),plan=JSON.parse(out(resp));S.lab.recipe=plan;Object.assign(S.lab,plan.shape||{});syncLabControls();drawRadio();buildTunehounds(text,plan);status("TUNEHOUNDS",plan.note||"forage ready")}catch(e){status("TUNEHOUNDS",`local forage · ${e.message}`)}}

function copySliceBuffer(buffer,a,z,reverse=false){let lo=clamp(Math.min(a,z),0,1),hi=clamp(Math.max(a,z),0,1);a=lo;z=hi;if(z-a<.005)z=Math.min(1,a+.005);let start=Math.floor(a*buffer.length),end=Math.max(start+1,Math.floor(z*buffer.length)),len=end-start,c=new AudioBuffer({length:len,numberOfChannels:buffer.numberOfChannels,sampleRate:buffer.sampleRate});for(let ch=0;ch<buffer.numberOfChannels;ch++){let src=buffer.getChannelData(ch),dst=c.getChannelData(ch);if(reverse){for(let i=0;i<len;i++)dst[i]=src[end-1-i]}else dst.set(src.subarray(start,end))}return c}
function envelopeCurve(points,n=512){let curve=new Float32Array(n);curve.fill(1);if(!points||points.length<2)return curve;let pts=[...points].sort((a,b)=>a[0]-b[0]);for(let i=0;i<n;i++){let x=i/(n-1),j=0;while(j<pts.length-2&&pts[j+1][0]<x)j++;let p0=pts[j],p1=pts[Math.min(j+1,pts.length-1)],u=p1[0]===p0[0]?0:(x-p0[0])/(p1[0]-p0[0]),y=p0[1]+(p1[1]-p0[1])*clamp(u,0,1);curve[i]=clamp(1-y,0.02,1)}return curve}
function driveCurveV9(amount){let n=2048,c=new Float32Array(n),k=+amount||0;for(let i=0;i<n;i++){let x=i*2/n-1;c[i]=k?((1+k)*x)/(1+k*Math.abs(x)):x}return c}
function grainify(buffer,amount){if(amount<.05)return buffer;let size=Math.max(64,Math.floor(buffer.sampleRate*(.18-(amount*.15)))),out=new AudioBuffer({length:buffer.length,numberOfChannels:buffer.numberOfChannels,sampleRate:buffer.sampleRate});for(let ch=0;ch<buffer.numberOfChannels;ch++){let src=buffer.getChannelData(ch),dst=out.getChannelData(ch);for(let i=0;i<dst.length;i++){let g=Math.floor(i/size),local=i%size,srcBase=(g%2===0?g:Math.max(0,g-1))*size,idx=Math.min(src.length-1,srcBase+local);let win=.5-.5*Math.cos(2*Math.PI*local/Math.max(1,size-1));dst[i]=src[idx]*(amount>.55?win:1)}}return out}
async function bounceLab(){let src=S.lab.buffer;if(!src){labSetState("FAILED","no decoded source");return null}labSetState("BOUNCING","offline render");busySoundLab(true);let cut=copySliceBuffer(src,S.lab.sel[0],S.lab.sel[1],S.lab.reverse),rate=clamp(+S.lab.rate,.25,2),dur=cut.duration/rate,frames=Math.max(128,Math.ceil(dur*cut.sampleRate)),off=new OfflineAudioContext(cut.numberOfChannels,frames,cut.sampleRate),bs=off.createBufferSource(),filter=off.createBiquadFilter(),shape=off.createWaveShaper(),gain=off.createGain(),delay=off.createDelay(1),wet=off.createGain(),dry=off.createGain(),merge=off.createGain();bs.buffer=cut;bs.playbackRate.value=rate;let tone=+S.lab.tone;if(tone<-.03){filter.type="lowpass";filter.frequency.value=18000*Math.pow(.16,-tone)}else if(tone>.03){filter.type="highpass";filter.frequency.value=70*Math.pow(45,tone)}else{filter.type="allpass";filter.frequency.value=1000}shape.curve=driveCurveV9(S.lab.drive);shape.oversample="2x";gain.gain.setValueCurveAtTime(envelopeCurve(S.lab.draw),0,Math.max(.01,dur));dry.gain.value=1;wet.gain.value=clamp(+S.lab.space,0,1)*.48;delay.delayTime.value=.16+.38*clamp(+S.lab.space,0,1);bs.connect(filter).connect(shape).connect(gain);gain.connect(dry).connect(merge);gain.connect(delay).connect(wet).connect(merge);merge.connect(off.destination);bs.start(0);let rendered=await off.startRendering();rendered=grainify(rendered,+S.lab.grain);S.lab.bounced=rendered;let t=labTrack();t.localBuffer=rendered;t.localName=`BOUNCE · ${S.lab.sourceName||t.label}`;t.sample=null;t.drums=false;t.label=t.localName;t.tag=`Radiodraw · ${[S.lab.reverse?"reverse":"",S.lab.rate!==1?`×${S.lab.rate.toFixed(2)}`:"",S.lab.grain?"grain":""].filter(Boolean).join(" ")||"shape"}`;labSetState("BOUNCED",`${rendered.duration.toFixed(2)}s local`);busySoundLab(false);render();drawRadio();labTrace();return rendered}
function busySoundLab(v){let b=$("#bounceBtn");if(b){b.disabled=v;b.textContent=v?"… BOUNCE":"BOUNCE"}}

async function auditionLab(){let b=S.lab.bounced||S.lab.buffer;if(!b)return;await ensureCtxV83();if(S.lab.audition){try{S.lab.audition.stop()}catch{}}let src=S.ctx.createBufferSource(),g=S.ctx.createGain();src.buffer=b;g.gain.value=.72;src.connect(g).connect(S.mix?.input||S.ctx.destination);src.start();S.lab.audition=src;labSetState("AUDITIONED",S.lab.bounced?"bounced":"source")}
async function importLab(file){if(!file)return;await ensureCtxV83();labSetState("FETCHING",file.name);try{let a=await file.arrayBuffer(),b=await S.ctx.decodeAudioData(a);S.lab.source={name:file.name,path:null,local:true};S.lab.sourceName=file.name;S.lab.buffer=b;S.lab.bounced=null;$("#labName").textContent=file.name;resetLabShape();drawRadio();labSetState("READY","imported local");buildTunehounds(file.name)}catch(e){labSetState("FAILED",e.message)}}

/* Playback with local bounced sound objects and no silent sample substitution. */
async function playV9(){
  if(S.playing){stopV9();return}await ensureCtxV83();let w=S.worlds[S.world];if(!w)return;configureMixV83(w);$("#playBtn").textContent="… LOAD";$("#playBtn").disabled=true;status("LOAD","checking sound manifest");
  let programs=w.tracks.filter(t=>!t.drums&&!t.sample&&!t.localBuffer).map(t=>t.program),sampleTracks=w.tracks.filter(t=>t.sample),failed=[];await Promise.all([preloadProgramsV83(programs),(async()=>{for(let t of sampleTracks){let b=await loadESBufV83(t.sample);t.sampleState=b?"READY":"FAILED";if(!b)failed.push(t)}})()]);
  let ready=w.tracks.length-failed.length;status("SOUND",`${ready} READY${failed.length?` · ${failed.length} FAILED`:""}`);let spb=60/w.bpm,beats=w.bars*w.meter[0]*(4/w.meter[1]);S.playing=true;S.start=S.ctx.currentTime+.06;S.dur=beats*spb;$("#playBtn").disabled=false;$("#playBtn").textContent="■ STOP";
  for(let t of w.tracks)for(let n of t.notes){let when=S.start+n.beat*spb,dur=n.dur*spb;if(t.drums){drumV83(n,when);continue}if(t.localBuffer){let src=S.ctx.createBufferSource(),g=S.ctx.createGain();src.buffer=t.localBuffer;g.gain.value=Math.max(.08,n.vel/127*.7);src.connect(g).connect(S.mix.input);src.start(when,0,Math.min(t.localBuffer.duration,dur));S.nodes.push(src,g);continue}if(t.sample){let b=S.esBuf.get(t.sample.name);if(!b)continue;let src=S.ctx.createBufferSource(),g=S.ctx.createGain();src.buffer=b;g.gain.value=Math.max(.08,n.vel/127*.58);src.connect(g).connect(S.mix.input);src.start(when,0,Math.min(b.duration,dur));S.nodes.push(src,g);continue}if(!scheduleSFV83(t,n,when,dur))synthV83(t,n,when,dur)}animateV9()}
function animateV9(){let w=S.worlds[S.world],spb=60/w.bpm,tick=()=>{if(!S.playing)return;let b=(S.ctx.currentTime-S.start)/spb;if(b>=S.dur/spb){stopV9();return}$("#timelineInner").style.setProperty("--pos",Math.max(0,b));$$('.hit').forEach(x=>{let t=w.tracks[+x.dataset.track],n=t.notes[+x.dataset.note];x.classList.toggle("playing",b>=n.beat&&b<n.beat+n.dur)});S.raf=requestAnimationFrame(tick)};tick()}
function stopV9(){if(!S.playing)return;S.playing=false;cancelAnimationFrame(S.raf);S.nodes.forEach(n=>{try{n.stop?.()}catch{}try{n.disconnect?.()}catch{}});S.nodes=[];$("#playBtn").disabled=false;$("#playBtn").textContent="▶ PLAY";$("#timelineInner").style.setProperty("--pos",0);$$('.hit').forEach(x=>x.classList.remove("playing"));status("READY","stopped")}

/* Extend trace so bounced samples identify their lineage. */
const renderTraceV8=renderTrace;
renderTrace=function(){renderTraceV8();let w=S.worlds[S.world],t=S.sel?w.tracks[S.sel.track]:null;if(t?.localBuffer){$("#causal").innerHTML+=cause("SOUND OBJECT",`${t.localName||t.label} · local bounce · ${t.localBuffer.duration.toFixed(2)}s`)+cause("SCULPT",t.tag||"Radiodraw")}}

/* Wire v9 after the existing application has initialized. */
$("#sculptBtn").onclick=()=>openSoundLab(null,false);
$("#forageBtn").onclick=()=>openSoundLab(null,true);
$("#sniffBtn").onclick=sniff;
$("#houndPrompt").onkeydown=e=>{if(e.key==="Enter")sniff()};
$$('[data-radio]').forEach(b=>b.onclick=()=>{S.lab.mode=b.dataset.radio;$$('[data-radio]').forEach(x=>x.classList.toggle("on",x===b));drawRadio()});
$("#clearDraw").onclick=()=>{S.lab.draw=[];drawRadio();labTrace()};
for(let [id,key] of [["rateCtl","rate"],["toneCtl","tone"],["driveCtl","drive"],["spaceCtl","space"],["grainCtl","grain"]]){$("#"+id).oninput=e=>{S.lab[key]=+e.target.value;syncLabControls();drawRadio()}}
$("#reverseCtl").onclick=()=>{S.lab.reverse=!S.lab.reverse;syncLabControls();drawRadio()};
$("#auditionBtn").onclick=auditionLab;$("#bounceBtn").onclick=bounceLab;$("#importBtn").onclick=()=>$("#importFile").click();$("#importFile").onchange=e=>importLab(e.target.files?.[0]);
let rc=$("#radioCanvas");rc.onpointerdown=radioDownFn;rc.onpointermove=radioMoveFn;rc.onpointerup=radioUpFn;rc.onpointercancel=radioUpFn;
window.addEventListener("resize",()=>{if($("#soundLabSheet").classList.contains("open"))drawRadio()});
$("#playBtn").onclick=playV9;

/* The library remains a picker. Long-press-like double tap on a result opens it in the lab after selection. */
const openSoundV8=openSound;
openSound=function(ti){openSoundV8(ti);S.lab.track=ti};

/* When EarSketch arrives, Tunehounds immediately become useful. */
const loadESV8=loadES;
loadES=async function(){await loadESV8();if($("#soundLabSheet").classList.contains("open"))buildTunehounds($("#houndPrompt").value||"")};

syncLabControls();
