"use strict";
/* TRACE v8.4 release gate: scope bounds, executable orchestration patches, startup assertions. */
(()=>{
  const priorCompose=globalThis.compose;
  const priorVariant=globalThis.v8PlanVariant;
  const priorSourcePatch=globalThis.sourcePatch;
  const priorInit=globalThis.v8Init;

  function boundRepeats(start,step,duration,end,max){
    start=+start;step=Math.max(.125,+step||.125);duration=Math.max(.05,+duration||.05);
    const last=end-duration-.0001;
    if(start>last)return 0;
    return Math.max(1,Math.min(max,Math.floor((last-start)/step)+1));
  }
  function boundPlan(plan){
    const p=clone(plan),s=scopeObject(),songEnd=beats(state.composition),lo=Math.max(0,s.start),hi=Math.min(songEnd,s.end);
    p.clear_ranges=(p.clear_ranges||[]).map(x=>({...x,start:clamp(+x.start,lo,hi),end:clamp(+x.end,lo,hi)})).filter(x=>x.end>x.start);
    p.note_loops=(p.note_loops||[]).map(x=>{
      const y={...x};y.start=clamp(+y.start,lo,Math.max(lo,hi-.05));y.step=Math.max(.125,+y.step||.125);y.duration=Math.min(Math.max(.05,+y.duration||.05),Math.max(.05,hi-y.start));
      y.repeats=boundRepeats(y.start,y.step,y.duration,hi,Math.min(V7_NOTE_LOOP_LIMIT*8,64));return y;
    }).filter(x=>x.repeats>0);
    p.drum_loops=(p.drum_loops||[]).map(x=>{const y={...x};y.start=clamp(+y.start,lo,Math.max(lo,hi-.12));y.step=Math.max(.125,+y.step||.125);y.repeats=boundRepeats(y.start,y.step,.12,hi,64);return y}).filter(x=>x.repeats>0);
    return validatePlan(p);
  }

  globalThis.compose=async function(text){return boundPlan(await priorCompose(text))};
  globalThis.v8PlanVariant=function(text,plan,degree){return boundPlan(priorVariant(text,boundPlan(plan),degree))};
  globalThis.sourcePatch=function(plan){
    const p=boundPlan(plan),raw=priorSourcePatch(p).split(/\r?\n/),out=[];let lastTrack=null;
    for(const line of raw){
      const tm=line.match(/^track\((\d+),/);if(tm)lastTrack=+tm[1];
      out.push(line);
      if(/^# GM \d+ · /.test(line)&&lastTrack!==null){const x=(p.track_changes||[]).find(v=>+v.track===lastTrack);if(x&&Number.isInteger(x.program))out.push(`instrument(${lastTrack}, ${clamp(x.program,0,127)})`)}
    }
    return out.join("\n");
  };

  function releaseAssert(ok,msg){if(!ok)throw Error(`TRACE RELEASE ASSERT · ${msg}`)}
  function selfTest(){
    releaseAssert(Array.isArray(V8_GM_NAMES)&&V8_GM_NAMES.length===128,"GM names must be 128");
    releaseAssert(Array.isArray(V8_GM_SLUGS)&&V8_GM_SLUGS.length===128,"GM slugs must be 128");
    releaseAssert(typeof TRACE_V7_BASE?.boot==="function","v7 boot seam missing");
    releaseAssert(typeof globalThis.v8PlanVariant==="function","branch generator missing");
    releaseAssert(typeof globalThis.sourcePatch==="function","source patch missing");
    for(const id of ["playBtn","stopBtn","sendBtn","trailBtn","timeline","messages"])releaseAssert(!!byId(id),`critical surface #${id} missing`);
    const probe={summary:"probe",interpretation:[],dimensions:[],alternatives:[],tempo:null,track_changes:[{track:1,name:"PROBE",kind:"lead",wave:"square",program:80,reason:"probe"}],clear_ranges:[],note_loops:[],drum_loops:[],effects:[],sample_assignments:[]};
    const patch=globalThis.sourcePatch(probe);releaseAssert(/instrument\(1, 80\)/.test(patch),"instrument identity is not executable source");
    return true;
  }
  globalThis.TRACE_RELEASE_SELF_TEST=selfTest;
  globalThis.v8Init=function(){selfTest();priorInit();document.documentElement.dataset.traceRelease="8.4"};
})();
