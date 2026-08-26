"use strict";
(async()=>{
  for(const src of ["core.js","ai.js","audio.js","ui-render.js","ui-interact.js","ui-tools.js","ui-boot.js"]){
    await new Promise((resolve,reject)=>{
      const s=document.createElement("script");
      s.src=src;
      s.onload=resolve;
      s.onerror=()=>reject(new Error("TRACE failed to load "+src));
      document.head.appendChild(s);
    });
  }
})().catch(err=>{console.error(err);document.body.innerHTML=`<pre style="padding:20px">${err.message}</pre>`});
