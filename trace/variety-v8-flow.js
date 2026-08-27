"use strict";
/* Three visible candidate paths. No hidden recursive wrappers. */
function previewMessage(id){
  const m=state.messages.find(x=>x.id===id&&x.plan);if(!m)return;
  const plan=m.variants?.[m.variantIndex||0]||m.plan;
  state.pending={...m,plan};
  state.preview=applyPlan(state.composition,plan,m.promptId,true);
  setStatus(`PREVIEW ${plan.variantLabel||""}`.trim(),true);
  renderAll();
  setTimeout(()=>setStatus("READY",false),500);
  if(innerWidth<=820)setMobile("world");
  log("preview",{messageId:id,promptId:m.promptId,variant:plan.variantLabel||"ROOT"});
}
function applyMessage(id){
  const m=state.messages.find(x=>x.id===id&&x.plan);if(!m)return;
  const plan=m.variants?.[m.variantIndex||0]||m.plan;
  if(!state.preview||state.pending?.id!==m.id||state.pending?.plan?.variantLabel!==plan.variantLabel)state.preview=applyPlan(state.composition,plan,m.promptId,true);
  for(const e of state.preview.events)delete e.preview;
  state.composition=clone(state.preview);state.preview=null;state.pending=null;
  syncSource();
  snapshot(`accepted ${plan.variantLabel||""}`.trim(),state.messages.find(x=>x.promptId===m.promptId&&x.role==="user")?.text||"",{dimensions:plan.dimensions||[],variant:plan.variantLabel||"ROOT",prediction:m.prediction||null});
  state.observer=m.observer||state.observer;
  setStatus("WRITTEN",true);renderAll();setTimeout(()=>setStatus("READY",false),650);
  showCoach("REFLECT",`You kept ${plan.variantLabel||"this path"}. The other branches remain alternative executable theories.`);
  log("apply",{messageId:id,promptId:m.promptId,variant:plan.variantLabel||"ROOT",dimensions:plan.dimensions||[]});
}
async function sendPrompt(){
  const input=byId("prompt");let text=input.value.trim();
  if(!text&&sketchPoints.length>=2)text="Use this sketch as a musical contour.";
  if(!text||state.busy)return;
  const promptId=nextPromptId++;addMsg("user",text,{promptId});input.value="";startLearningWait(text);setStatus("READING",true);byId("sendBtn").disabled=true;
  try{
    await Promise.allSettled([v7LoadTags(),v8LoadEarSketch()]);
    const root=await compose(text),variants=[root,v8PlanVariant(text,root,1),v8PlanVariant(text,root,2)];
    finishLearningWait(root);setStatus("BUILDING 3 PATHS",true);
    const patch=sourcePatch(root),observer=await observe(text,root,patch);state.observer=observer;
    addMsg("assistant",root.summary,{promptId,plan:root,variants,variantIndex:0,patch,observer,prediction:state.learning.prediction});
    log("plan",{promptId,dimensions:root.dimensions,summary:root.summary,variants:variants.map(x=>x.variantLabel)});setStatus("READY",false);
  }catch(e){
    state.learning.active=false;renderLearningWait();addMsg("error",e.message||String(e));setStatus("ERROR",false);
    showCoach("FAILURE IS EVIDENCE","No patch was applied. The failure belongs to the model/request seam, not to your song.");
  }finally{state.busy=false;byId("sendBtn").disabled=false}
}
