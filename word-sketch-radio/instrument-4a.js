'use strict';

let latestCritique='';
let chatAttachments=[];
let chatTurns=[];
let lastGuideReply='';
let provenance={nodes:[],edges:[]};
let generations=[];
let currentGenerationId=null;
let historyPage=0;
let traceCounter=0;
const STORE_KEY='word-sketch-song-history-v3';
const FLOW_LANES=['word','sketch','pipeline','code','song','spectrum','chat'];

function nowId(prefix='n'){return `${prefix}-${Date.now().toString(36)}-${(++traceCounter).toString(36)}`}
function smallHash(s){s=String(s??'');let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return (h>>>0).toString(36)}
function thumb(canvas,w=260,h=150,bg='#fffdf4'){try{const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d');x.fillStyle=bg;x.fillRect(0,0,w,h);x.drawImage(canvas,0,0,w,h);return c.toDataURL('image/jpeg',.64)}catch{return''}}
function artifactPayload(type){if(type==='word')return $('#wordInput').value.trim();if(type==='sketch')return thumb(sketch,240,180,'#fff');if(type==='spectrum')return song?canvasDataURL(specInk,specBase):'';if(type==='song')return song?JSON.stringify(song):'';if(type==='code')return songToPython();if(type==='pipeline')return pipeline?JSON.stringify(pipeline):'';if(type==='flow')return JSON.stringify({nodes:provenance.nodes.slice(-36),edges:provenance.edges.slice(-48)});if(type==='chat')return latestCritique;return''}
function traceNode(type,label,payload=''){const node={id:nowId(type[0]||'n'),type,label:String(label||type).slice(0,64),hash:smallHash(payload),time:Date.now(),generation:currentGenerationId};provenance.nodes.push(node);if(provenance.nodes.length>120)provenance.nodes=provenance.nodes.slice(-120);return node.id}
function traceEdge(from,to,op){if(!from||!to)return;provenance.edges.push({id:nowId('e'),from,to,op,time:Date.now()});if(provenance.edges.length>180)provenance.edges=provenance.edges.slice(-180)}
function newestNode(type){for(let i=provenance.nodes.length-1;i>=0;i--)if(provenance.nodes[i].type===type)return provenance.nodes[i];return null}
function snapshotNode(type,label=type.toUpperCase()){const payload=artifactPayload(type);if(!payload)return null;const last=newestNode(type);if(last&&last.hash===smallHash(payload))return last.id;return traceNode(type,label,payload)}
function traceTransform(fromType,toType,op){const a=snapshotNode(fromType,fromType.toUpperCase()),b=snapshotNode(toType,toType.toUpperCase());traceEdge(a,b,op);persistTrace();refreshFlow()}

function generationPath(){const recent=provenance.edges.slice(-8).map(e=>e.op.replaceAll('→','>'));return recent.length?recent.join(' · '):'SOURCE > SONG'}
function recordGeneration(previousSong,previousPipeline,promptText=''){
  const parent=currentGenerationId;
  const id=nowId('g');
  currentGenerationId=id;
  const wordId=snapshotNode('word','WORD');
  const sketchId=hasSketchInk()?snapshotNode('sketch','SKETCH'):null;
  const spectrumInput=(previousSong&&dirty.spectrum)?snapshotNode('spectrum','SPECTRUM MARKS'):null;
  const chatId=latestCritique?snapshotNode('chat','CRITIQUE'):null;
  const priorSongNode=previousSong?newestNode('song')?.id:null;
  const pipeId=traceNode('pipeline','AGENTS',JSON.stringify(pipeline));
  const codeId=traceNode('code','CODE',songToPython());
  const songId=traceNode('song',song?.title||'SONG',JSON.stringify(song));
  const spectrumId=traceNode('spectrum','SPECTRUM',song?.title||'spectrum');
  for(const source of [wordId,sketchId,spectrumInput,chatId,priorSongNode])traceEdge(source,pipeId,source===priorSongNode?'LOOP':'FEED');
  traceEdge(pipeId,codeId,'ASSEMBLE');traceEdge(pipeId,songId,'RENDER');traceEdge(songId,spectrumId,'ANALYZE');
  const g={id,parent,time:Date.now(),title:song?.title||'SONG',word:$('#wordInput').value.trim(),critique:latestCritique,prompt:promptText,pipeline:structuredCloneSafe(pipeline),song:structuredCloneSafe(song),sketch:hasSketchInk()?thumb(sketch,320,240,'#fff'):'',spectrum:thumb(specInk,320,150,'#fffdf4'),path:generationPath()};
  generations.push(g);if(generations.length>24)generations=generations.slice(-24);
  latestCritique='';$('#changeInput').value='';
  persistTrace();renderHistory();refreshFlow();
}
function structuredCloneSafe(v){try{return JSON.parse(JSON.stringify(v))}catch{return null}}
function persistTrace(){try{localStorage.setItem(STORE_KEY,JSON.stringify({provenance,generations,currentGenerationId}))}catch{}}
function loadTrace(){try{const d=JSON.parse(localStorage.getItem(STORE_KEY)||'null');if(d){provenance=d.provenance||provenance;generations=d.generations||[];currentGenerationId=d.currentGenerationId||null}}catch{}renderHistory();refreshFlow()}

async function restoreGeneration(id){const g=generations.find(x=>x.id===id);if(!g)return;stop();pipeline=structuredCloneSafe(g.pipeline);song=structuredCloneSafe(g.song);currentGenerationId=g.id;$('#wordInput').value=g.word||'';fitWord();if(g.sketch)await loadSketchImage(g.sketch);else sk.clearRect(0,0,1024,1024);si.clearRect(0,0,1024,512);if(g.spectrum){const im=new Image();im.onload=()=>si.drawImage(im,0,0,1024,512);im.src=g.spectrum}if(song)await renderSong();resetDirty();setState('BRANCH READY','good');renderHistory();refreshFlow();if(innerWidth<=900)goColumn(1)}
function renderHistory(){const box=$('#historyGrid');if(!box)return;if(!generations.length){box.innerHTML='<div class="historyEmpty">NO RUNS</div>';return}const per=innerWidth<=900?4:6,pages=Math.max(1,Math.ceil(generations.length/per));historyPage=Math.max(0,Math.min(historyPage,pages-1));const ordered=[...generations].reverse(),slice=ordered.slice(historyPage*per,(historyPage+1)*per);box.innerHTML=slice.map(g=>`<button class="histCard ${g.id===currentGenerationId?'current':''}" data-id="${g.id}"><span class="histId">${new Date(g.time).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}${g.parent?' · BRANCH':''}</span><span class="histTitle">${escapeHtml(g.title)}</span><span class="histPath">${escapeHtml(g.path)}</span></button>`).join('');$$('.histCard').forEach(b=>b.onclick=()=>restoreGeneration(b.dataset.id));$('#histPrev').disabled=historyPage<=0;$('#histNext').disabled=historyPage>=pages-1}
$('#histPrev').onclick=()=>{historyPage--;renderHistory()};$('#histNext').onclick=()=>{historyPage++;renderHistory()};

function refreshFlow(){const c=$('#flowCanvas');if(!c)return;const x=c.getContext('2d'),W=c.width,H=c.height;x.clearRect(0,0,W,H);x.fillStyle='#fffdf4';x.fillRect(0,0,W,H);const nodes=provenance.nodes.slice(-42),ids=new Set(nodes.map(n=>n.id)),edges=provenance.edges.filter(e=>ids.has(e.from)&&ids.has(e.to));const laneY=t=>{const i=Math.max(0,FLOW_LANES.indexOf(t));return 34+i*(H-90)/(FLOW_LANES.length-1)};const minT=nodes[0]?.time||0,maxT=nodes.at(-1)?.time||minT+1;const pos=new Map();nodes.forEach((n,i)=>{const px=30+((n.time-minT)/(Math.max(1,maxT-minT)))*(W-60),py=laneY(n.type);pos.set(n.id,[px,py])});x.lineWidth=2;x.strokeStyle='rgba(11,12,11,.32)';for(const e of edges){const a=pos.get(e.from),b=pos.get(e.to);if(!a||!b)continue;x.beginPath();x.moveTo(...a);const mx=(a[0]+b[0])/2;x.bezierCurveTo(mx,a[1],mx,b[1],...b);x.stroke()}const curSong=newestNode('song')?.id;for(const n of nodes){const [px,py]=pos.get(n.id);x.fillStyle=n.id===curSong?'#ff4e26':n.generation===currentGenerationId?'#eaff52':'#0b0c0b';x.fillRect(px-5,py-5,10,10);if(n.id===curSong||nodes.length<18){x.font='700 13px ui-monospace,monospace';x.fillStyle='#0b0c0b';x.fillText(n.type.toUpperCase(),Math.min(W-80,px+8),py+4)}}const path=edges.slice(-6).map(e=>e.op).join(' · ');$('#flowPath').textContent=path||'WORD · SKETCH · CODE · SONG'}

function panelAttachment(name){if(name==='word')return{type:'word',label:'WORD',text:$('#wordInput').value.trim()||'(empty)'};if(name==='sketch')return{type:'sketch',label:'SKETCH',image:canvasDataURL(sketch)};if(name==='song')return{type:'song',label:'SONG',text:song?JSON.stringify(song):'(no song)'};if(name==='spectrum')return{type:'spectrum',label:'SPECTRUM',image:canvasDataURL(specInk,specBase)};if(name==='code')return{type:'code',label:'CODE',text:codeText()};if(name==='flow')return{type:'flow',label:'FLOW',text:JSON.stringify({nodes:provenance.nodes.slice(-36),edges:provenance.edges.slice(-48)})};return null}
function attachPanel(name){const a=panelAttachment(name);if(!a)return;chatAttachments.push(a);if(chatAttachments.length>5)chatAttachments.shift();renderAttachments();if(innerWidth<=900)goColumn(3)}
$$('.panelSend').forEach(b=>b.onclick=()=>attachPanel(b.dataset.panel));
function renderAttachments(){const box=$('#attachments');box.innerHTML=chatAttachments.map((a,i)=>`<div class="attachChip">${escapeHtml(a.label)}<button data-rm="${i}" aria-label="Remove">X</button></div>`).join('');$$('[data-rm]').forEach(b=>b.onclick=()=>{chatAttachments.splice(Number(b.dataset.rm),1);renderAttachments()})}
function currentLikelyPanel(){if(innerWidth<=900){const i=Math.round($('#stage').scrollLeft/innerWidth);if(i===0)return hasSketchInk()?'sketch':'word';return ['word','song','code','flow'][i]||'word'}return song?'song':hasSketchInk()?'sketch':'word'}