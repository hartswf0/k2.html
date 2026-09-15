/* <composition> is authoritative; <candidate> is separate. [validate] guards every [commit]. */
(function(root){root.PondCore=(function createPondCore() {
  const clone=x=>JSON.parse(JSON.stringify(x)),equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  function assert(c,m){if(!c)throw Error(m)}
  function note(beat,pitch,duration=.5,velocity=.5){return{beat,pitch,duration,velocity}}
  function starter(){return{version:1,tempo:96,bars:8,tracks:[
    {id:'drums',name:'Drums',kind:'drums',locked:false,bars:Array.from({length:8},()=>[note(0,36,.2,.8),note(1,38,.2),note(2,36,.2,.7),note(3,38,.2)])},
    {id:'bass',name:'Bass',kind:'bass',locked:false,bars:Array.from({length:8},(_,i)=>[note(0,[36,44,41,43][i%4],1.5),note(2,[36,44,41,43][i%4],1.5)])},
    {id:'keys',name:'Keys',kind:'keys',locked:false,bars:Array.from({length:8},(_,i)=>[0,3,7].map(n=>note(0,[60,56,53,55][i%4]+n,3.5,.3)))},
    {id:'lead',name:'Lead',kind:'lead',locked:false,bars:Array.from({length:8},(_,i)=>[note(.5,[72,75,77,74][i%4],.5,.35),note(2.5,[79,77,75,72][i%4],.75,.3)])}
  ],lyrics:Array.from({length:8},()=>({text:'',locked:false}))}}
  function notes(ns){assert(Array.isArray(ns)&&ns.length<=64,'At most 64 notes per bar.');for(const n of ns){assert(n&&Object.keys(n).sort().join(',')==='beat,duration,pitch,velocity','Notes require beat, pitch, duration and velocity only.');assert(Number.isFinite(n.beat)&&n.beat>=0&&n.beat<4,'Beat must be within 0–4.');assert(Number.isFinite(n.duration)&&n.duration>=.05&&n.beat+n.duration<=4.000001,'A note cannot cross its bar boundary.');assert(Number.isInteger(n.pitch)&&n.pitch>=24&&n.pitch<=96,'MIDI pitch must be 24–96.');assert(Number.isFinite(n.velocity)&&n.velocity>=0&&n.velocity<=1,'Velocity must be 0–1.');}}
  function validate(c){assert(c&&c.version===1,'Unsupported project.');assert(Number.isInteger(c.bars)&&c.bars>=4&&c.bars<=32,'Use 4–32 bars.');assert(Number.isInteger(c.tempo)&&c.tempo>=40&&c.tempo<=200,'Use 40–200 BPM.');assert(Array.isArray(c.tracks)&&c.tracks.length>=1&&c.tracks.length<=8,'Use 1–8 tracks.');assert(new Set(c.tracks.map(t=>t.id)).size===c.tracks.length,'Duplicate tracks.');for(const t of c.tracks){assert(typeof t.id==='string'&&t.id.length<=50&&typeof t.name==='string'&&t.name.length<=80,'Invalid track identity.');assert(['drums','bass','keys','lead'].includes(t.kind),'Invalid instrument.');assert(typeof t.locked==='boolean','Invalid lock.');assert(Array.isArray(t.bars)&&t.bars.length===c.bars,'Track length mismatch.');t.bars.forEach(notes);}assert(Array.isArray(c.lyrics)&&c.lyrics.length===c.bars,'Lyric length mismatch.');for(const l of c.lyrics)assert(l&&typeof l.text==='string'&&l.text.length<=500&&typeof l.locked==='boolean','Invalid lyric.');return clone(c)}
  function range(c,s){assert(Number.isInteger(s.start)&&Number.isInteger(s.end)&&s.start>=0&&s.end>=s.start&&s.end<c.bars,'Invalid selection.')}
  function passage(c,s){range(c,s);return{start:s.start,end:s.end,tracks:c.tracks.map(t=>({id:t.id,bars:clone(t.bars.slice(s.start,s.end+1))})),lyrics:c.lyrics.slice(s.start,s.end+1).map(l=>l.text),summary:'Edited passage'}}
  function proposal(raw,c,s,scope='both'){range(c,s);assert(['both','music','lyrics'].includes(scope),'Invalid scope.');assert(raw&&raw.start===s.start&&raw.end===s.end,'Candidate must match the selected bars.');const count=s.end-s.start+1;assert(Array.isArray(raw.tracks)&&raw.tracks.length===c.tracks.length,'Keep all existing tracks.');assert(new Set(raw.tracks.map(t=>t.id)).size===c.tracks.length,'Duplicate candidate tracks.');
    const tracks=c.tracks.map(t=>{const p=raw.tracks.find(x=>x.id===t.id);assert(p&&Array.isArray(p.bars)&&p.bars.length===count,'Track bars must match selection.');p.bars.forEach(notes);if(t.locked||scope==='lyrics')assert(equal(p.bars,t.bars.slice(s.start,s.end+1)),t.name+' is protected.');return{id:t.id,bars:clone(p.bars)}});
    assert(Array.isArray(raw.lyrics)&&raw.lyrics.length===count,'One lyric line per selected bar.');raw.lyrics.forEach((line,i)=>{assert(typeof line==='string'&&line.length<=500,'Invalid lyric line.');if(c.lyrics[s.start+i].locked||scope==='music')assert(line===c.lyrics[s.start+i].text,'A lyric line is protected.');});return{start:s.start,end:s.end,tracks,lyrics:clone(raw.lyrics),summary:typeof raw.summary==='string'?raw.summary.slice(0,1000):'Candidate passage'};
  }
  function apply(c,p){const next=clone(c);for(const t of next.tracks){p.tracks.find(x=>x.id===t.id).bars.forEach((bar,i)=>next.tracks.find(x=>x.id===t.id).bars[p.start+i]=clone(bar));}p.lyrics.forEach((text,i)=>next.lyrics[p.start+i].text=text);return next}
  function session(initial=starter()){
    let c=validate(initial),s={start:0,end:3},rev=0,cand=null,history=[];
    const snapshot=()=>clone({composition:c,selection:s,revision:rev,candidate:cand,history:history.map(h=>({label:h.label,at:h.at}))});
    function mutate(label,fn){const next=clone(c);fn(next);validate(next);history.push({label,at:new Date().toISOString(),before:clone(c)});history=history.slice(-30);c=next;rev++;cand=null;s={start:Math.min(s.start,c.bars-1),end:Math.min(s.end,c.bars-1)};return snapshot()}
    return{snapshot,select(start,end){range(c,{start,end});s={start,end};cand=null;return snapshot()},mutate,
      propose(raw,scope='both',baseRevision=rev,baseSelection=s){assert(baseRevision===rev&&equal(baseSelection,s),'Proposal expired.');cand={proposal:proposal(raw,c,s,scope),scope,revision:rev};return snapshot()},
      commit(){assert(cand,'No candidate.');assert(cand.revision===rev,'Candidate expired.');const p=proposal(cand.proposal,c,s,cand.scope);return mutate('Committed bars '+(s.start+1)+'–'+(s.end+1),n=>Object.assign(n,apply(c,p)))},
      discard(){cand=null;return snapshot()},undo(){assert(history.length,'Nothing to undo.');c=history.pop().before;rev++;cand=null;s={start:Math.min(s.start,c.bars-1),end:Math.min(s.end,c.bars-1)};return snapshot()},
      replace(next){const valid=validate(next);return mutate('Opened project',c=>Object.assign(c,valid))}
    }
  }
  return{clone,equal,assert,starter,validate,passage,proposal,apply,session};
})();})(globalThis);
