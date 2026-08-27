"use strict";
(async()=>{
  const version="8.4";
  const files=[
    "core.js","ai.js","audio.js","ui.js",
    "variety-core.js","variety-ai.js","variety-tags.js","variety-audio.js","variety-ui.js",
    "variety-v8-base.js","variety-v8-runtime.js","variety-v8-fixes.js","variety-v8-flow.js","variety-v8-finish.js","variety-v8-boot.js",
    "mobile-hotfix.js"
  ];
  for(const src of files){
    await new Promise((resolve,reject)=>{
      const s=document.createElement("script");s.src=`${src}?v=${version}`;s.onload=resolve;
      s.onerror=()=>reject(new Error("TRACE failed to load "+src));
      document.head.appendChild(s);
    });
  }
  boot();
})().catch(err=>{
  console.error(err);
  document.body.innerHTML=`<pre style="padding:20px;font:14px monospace">TRACE BOOT ERROR\n${String(err.message||err)}</pre>`;
});
