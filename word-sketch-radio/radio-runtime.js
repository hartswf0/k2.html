'use strict';

// Make visible relation patches affect playback and the world, not only the prompt.
const sketchradioBaseScheduleTone=scheduleTone;
const sketchradioBaseDrawRadioCursor=drawRadioCursor;

function radioEnergyAtBeat(beat=currentBeat){
  const a=radioVoice.analysis;
  if(!a?.envelope?.length)return 0;
  const p=Math.max(0,Math.min(.999999,Number(beat||0)/48));
  return a.envelope[Math.min(a.envelope.length-1,Math.floor(p*a.envelope.length))]||0;
}
function relationTargetsScore(r){
  const t=String(r?.to||'').toUpperCase();
  return /SCORE|MUSIC|HARMONY|RHYTHM|TIMBRE|STRINGS|DRUM|BASS/.test(t);
}
function relationScoreLag(){
  const rel=(radioRelations||[]).filter(relationTargetsScore);
  if(!rel.length)return 0;
  let n=0,d=0;
  for(const r of rel){const w=Math.max(.05,Number(r.strength)||0);n+=(Number(r.lag_seconds)||0)*w;d+=w}
  return Math.max(-3,Math.min(4,n/Math.max(.001,d)));
}
function relationSilenceWeight(){
  let w=0;
  for(const r of radioRelations||[]){
    if(!relationTargetsScore(r))continue;
    const f=String(r.from||'').toUpperCase();
    if(/SILENCE|PAUSE|BREATH|PHRASE END/.test(f))w=Math.max(w,Number(r.strength)||0);
  }
  return w;
}
function relationVoiceWeight(){
  let w=0;
  for(const r of radioRelations||[]){
    if(!relationTargetsScore(r))continue;
    const f=String(r.from||'').toUpperCase();
    if(/VOICE|PERFORMANCE|POEM/.test(f))w=Math.max(w,Number(r.strength)||0);
  }
  return w;
}
function performanceVelocity(ev){
  const energy=radioEnergyAtBeat(ev.offset||0),base=Math.max(1,Number(ev.velocity)||70);
  const silence=relationSilenceWeight(),voice=relationVoiceWeight();
  let factor=1;
  if(radioMode==='ANSWER')factor=.12+(1-energy)*.98;
  else if(radioMode==='HARMONIZE')factor=.42+(1-energy)*.58;
  else if(radioMode==='SHADOW')factor=.30+energy*.70;
  else if(radioMode==='COUNTERPOINT')factor=energy>.78?.48:.82;
  else if(radioMode==='RESIST')factor=.86;
  else if(radioMode==='EMBODY')factor=.30+(1-energy)*.48;
  if(silence)factor*=.55+(1-energy)*silence*.75;
  if(voice&&!silence&&radioMode!=='ANSWER')factor*=.72+energy*voice*.35;
  return Math.max(2,Math.min(127,Math.round(base*factor)));
}

scheduleTone=function(ctx,ev,when,dur){
  const lag=relationScoreLag();
  const shifted=Math.max(ctx.currentTime+.006,when+lag);
  const shaped={...ev,velocity:performanceVelocity(ev)};
  if(radioMode==='ANSWER'&&radioVoice.analysis&&radioEnergyAtBeat(ev.offset||0)>.82&&relationSilenceWeight()>.35)return;
  return sketchradioBaseScheduleTone(ctx,shaped,shifted,dur);
};

drawRadioCursor=function(){
  sketchradioBaseDrawRadioCursor();
  const c=$('#radioWorldCursor');if(!c)return;
  const x=c.getContext('2d'),w=c.width,h=c.height,energy=radioEnergyAtBeat();
  const horizon=h*(.72-energy*.28);
  x.save();
  x.strokeStyle='rgba(8,10,8,.72)';x.lineWidth=2;
  x.beginPath();x.moveTo(0,horizon);x.lineTo(w,horizon);x.stroke();
  x.fillStyle='rgba(8,10,8,.055)';x.fillRect(0,horizon,w,h-horizon);

  const skyActive=(radioRelations||[]).some(r=>/SKY|LIGHT|CLOUD|WEATHER/.test(String(r.to||'').toUpperCase()));
  if(skyActive){
    x.strokeStyle='rgba(8,10,8,.18)';x.lineWidth=1.5;
    const drift=(currentBeat/48)*w;
    for(let i=0;i<4;i++){
      const y=42+i*34+energy*18;
      x.beginPath();x.moveTo((drift+i*170)%w-120,y);x.bezierCurveTo(w*.28,y-22*energy,w*.58,y+18*energy,w+40,y-5);x.stroke();
    }
  }
  const landActive=(radioRelations||[]).some(r=>/LAND|GROUND|TERRAIN/.test(String(r.to||'').toUpperCase()));
  if(landActive){
    x.strokeStyle='rgba(8,10,8,.22)';
    for(let i=1;i<=3;i++){const y=horizon+i*(h-horizon)/4;x.beginPath();x.moveTo(0,y);x.lineTo(w,y+Math.sin(currentBeat*.18+i)*9*energy);x.stroke()}
  }
  x.restore();
};

// Make the live patch state legible in the transport without another panel.
const sketchradioBaseRenderRelations=renderRadioRelations;
renderRadioRelations=function(summary=''){
  const lag=relationScoreLag();
  const suffix=radioRelations?.length?` · SCORE ${lag>=0?'+':''}${lag.toFixed(1)}s`:'';
  sketchradioBaseRenderRelations((summary||`${radioRelations.length||0} ACTIVE`)+suffix);
  drawRadioCursor();
};

renderRadioRelations('RELATION RUNTIME READY');
