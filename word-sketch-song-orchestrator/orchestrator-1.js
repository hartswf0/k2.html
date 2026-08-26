'use strict';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const API_PROXY='https://lcxykqgddnekrimawpie.supabase.co/functions/v1/openai-proxy';
const SUPABASE_PUBLISHABLE='sb_publishable_bF2QF1eJmzhi4Pj-dxE-0g_mnm6zRSh';
let apiKey='', verified=false, orchestration=null, programSong=null, selectedRule=-1, activePanel='code';
let audioCtx=null, scheduled=[], playing=false, playStartTime=0, playStartBeat=0, currentBeat=0, raf=0;
let sketchDirty=false, spectrumDirty=false, drawingSketch=false, drawingSpectrum=false, lastPt=null;
let flowEvents=[], snapshots=[], branchId=0, chatImageData='';
const chats={word:[],sketch:[],theory:[],code:[],song:[],spectrum:[]};
const sketch=$('#sketch'), sk=sketch.getContext('2d'), spectrumBase=$('#spectrumBase'), spb=spectrumBase.getContext('2d'), spectrumInk=$('#spectrumInk'), spi=spectrumInk.getContext('2d'), songViz=$('#songViz'), sv=songViz.getContext('2d');

const NOTE_SCHEMA={type:'object',additionalProperties:false,required:['pitch','offset','duration','velocity'],properties:{pitch:{type:'string',pattern:'^[A-G][#b]?[0-8]$'},offset:{type:'number',minimum:0,maximum:15.99},duration:{type:'number',exclusiveMinimum:0,maximum:8},velocity:{type:'integer',minimum:1,maximum:127}}};
const TRACK_SCHEMA={type:'object',additionalProperties:false,required:['name','program','notes'],properties:{name:{type:'string'},program:{type:'integer',minimum:0,maximum:127},notes:{type:'array',minItems:1,maxItems:64,items:NOTE_SCHEMA}}};
const SECTION_SCHEMA={type:'object',additionalProperties:false,required:['name','tracks'],properties:{name:{type:'string',enum:['A1','B','A2']},tracks:{type:'array',minItems:1,maxItems:5,items:TRACK_SCHEMA}}};
const SONG_SCHEMA={type:'object',additionalProperties:false,required:['title','tempo','meter','sections'],properties:{title:{type:'string'},tempo:{type:'integer',minimum:40,maximum:220},meter:{type:'string'},sections:{type:'array',minItems:3,maxItems:3,items:SECTION_SCHEMA}}};
const SOURCE_SCHEMA={type:'object',additionalProperties:false,required:['title','reading','evidence'],properties:{title:{type:'string'},reading:{type:'string'},evidence:{type:'array',minItems:2,maxItems:6,items:{type:'object',additionalProperties:false,required:['kind','observation','musical_hint'],properties:{kind:{type:'string'},observation:{type:'string'},musical_hint:{type:'string'}}}}}};
const THEORY_SCHEMA={type:'object',additionalProperties:false,required:['rules','sections'],properties:{rules:{type:'array',minItems:5,maxItems:5,items:{type:'object',additionalProperties:false,required:['from_feature','operation','to_feature','rationale','confidence'],properties:{from_feature:{type:'string'},operation:{type:'string'},to_feature:{type:'string'},rationale:{type:'string'},confidence:{type:'number',minimum:0,maximum:1}}}},sections:{type:'array',minItems:3,maxItems:3,items:{type:'object',additionalProperties:false,required:['name','function','energy'],properties:{name:{type:'string',enum:['A1','B','A2']},function:{type:'string'},energy:{type:'number',minimum:0,maximum:1}}}}}};
const LISTENER_SCHEMA={type:'object',additionalProperties:false,required:['hearing','tensions','next_question'],properties:{hearing:{type:'string'},tensions:{type:'array',minItems:1,maxItems:4,items:{type:'string'}},next_question:{type:'string'}}};
const CHAT_SCHEMA={type:'object',additionalProperties:false,required:['answer','replacement_program'],properties:{answer:{type:'string'},replacement_program:{type:'string'}}};

function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function setState(text,kind=''){ $('#stateText').textContent=text; $('#stateDot').className=kind; }
function panelReady(id,on=true){const p=$('#'+id+'Panel');if(p){p.classList.toggle('ready',on);p.classList.toggle('changed',false)}}
function panelChanged(id,on=true){const p=$('#'+id+'Panel');if(p){p.classList.toggle('changed',on);if(on)p.classList.remove('ready')}}
function gateState(text,kind=''){const el=$('#gateState');el.className='gateState '+kind;el.innerHTML=`<span></span>${esc(text)}`}
function agentState(name,state=''){const b=$(`[data-agent="${name}"]`);if(b)b.className=state}
function resetAgentStates(){$$('#buildPath button').forEach(b=>b.className='')}
function busy(on,label='READ',pct=10){$('#busyLayer').classList.toggle('on',on);$('#busyText').textContent=label;$('#busyLayer .busyBar span').style.width=pct+'%';if(on)setState(label,'busy')}
function updateMake(){const hasWord=$('#wordInput').value.trim().length>0;$('#makeBtn').disabled=!verified||(!programSong&&!hasWord&&!sketchDirty);$('#makeLabel').textContent=programSong?'LOOP':'MAKE'}
function syncModels(){ $('#textModel').value=$('#gateModel').value;$('#imageModel').value=$('#gateImage').value }

async function apiFetch(path,opts={}){
  let body=null;if(opts.body){try{body=typeof opts.body==='string'?JSON.parse(opts.body):opts.body}catch{body=opts.body}}
  const r=await fetch(API_PROXY,{method:'POST',headers:{Authorization:`Bearer ${SUPABASE_PUBLISHABLE}`,apikey:SUPABASE_PUBLISHABLE,'Content-Type':'application/json','x-openai-key':apiKey},body:JSON.stringify({path,method:opts.method||'GET',body})});
  let data=null;try{data=await r.json()}catch{}if(!r.ok){const msg=typeof data?.error==='string'?data.error:data?.error?.message||data?.message||`${r.status} ${r.statusText}`;throw new Error(msg)}return data
}
function responseText(data){if(data.output_text)return data.output_text;for(const item of data.output||[])for(const c of item.content||[]){if(c.type==='output_text'&&c.text)return c.text;if(typeof c.text==='string')return c.text}throw new Error('No response text')}
async function structured(content,name,schema){const data=await apiFetch('/responses',{method:'POST',body:{model:$('#textModel').value,input:[{role:'user',content}],text:{format:{type:'json_schema',name,schema,strict:true}}}});return JSON.parse(responseText(data))}
async function plainResponse(content){const data=await apiFetch('/responses',{method:'POST',body:{model:$('#textModel').value,input:[{role:'user',content}]}});return responseText(data)}
function canvasURL(canvas,bg=null){if(!bg)return canvas.toDataURL('image/png');const c=document.createElement('canvas');c.width=bg.width;c.height=bg.height;const x=c.getContext('2d');x.drawImage(bg,0,0);x.drawImage(canvas,0,0);return c.toDataURL('image/png')}
function imageContent(prompt,items=[]){const c=[{type:'input_text',text:prompt}];for(const it of items){c.push({type:'input_text',text:it.label});c.push({type:'input_image',image_url:it.url,detail:'high'})}return c}
async function verify(){const key=$('#keyInput').value.trim()||apiKey;if(!key){gateState('ENTER KEY','bad');return}apiKey=key;$('#checkKey').disabled=true;gateState('CHECKING','busy');try{const data=await apiFetch('/models');const ids=new Set((data.data||[]).map(x=>x.id));verified=true;syncModels();gateState(ids.has($('#textModel').value)?'READY':'KEY OK / MODEL CHECK ON USE','good');$('#gate').classList.add('hidden');$('#app').setAttribute('aria-hidden','false');setState('READY','good');updateMake()}catch(e){verified=false;gateState(e.message,'bad')}finally{$('#checkKey').disabled=false}}
$('#checkKey').onclick=verify;$('#keyInput').addEventListener('keydown',e=>{if(e.key==='Enter')verify()});
