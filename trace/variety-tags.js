"use strict";
/* Full tag corpus, sharded so startup remains light. One bounded read on first generation. */
async function v7LoadTags(){
  if(v7Tags.length>=4000)return v7Tags;
  if(v7TagPromise)return v7TagPromise;
  const files=Array.from({length:8},(_,i)=>`tags/treblo-${String(i+1).padStart(2,"0")}.txt`);
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),7000);
  v7TagPromise=Promise.all(files.map(async path=>{
    const r=await fetch(path,{cache:"force-cache",signal:controller.signal});
    if(!r.ok)throw Error(`tag shard ${r.status}`);
    return r.text();
  })).then(parts=>{
    const tags=parts.join("\n").split(/\r?\n/).map(x=>x.trim().toLowerCase()).filter(x=>x&&x.length<96);
    v7Tags=[...new Set(tags)].slice(0,5000);
    if(v7Tags.length<4000)throw Error("tag corpus incomplete");
    return v7Tags;
  }).catch(()=>{
    v7Tags=[...V7_COMMON_TAGS];
    return v7Tags;
  }).finally(()=>clearTimeout(timer));
  return v7TagPromise;
}
