'use strict';

// Use the documented Chat Completions audio-input path for HEAR.
const SKETCHRADIO_AUDIO_CHAT_MODEL='gpt-audio-1.5';
const deepResponsesHear=describeCurrentSound;

function chatMessageText(message){
  const c=message?.content;
  if(typeof c==='string')return c;
  if(Array.isArray(c))return c.map(p=>typeof p==='string'?p:(p?.text||p?.content||'')).filter(Boolean).join('\n');
  return '';
}

describeCurrentSound=async function(){
  if(!radioVoice.buffer)return;
  if(!verified){$('#gate').classList.remove('hidden');return}
  busy(true,'HEARING RAW AUDIO');
  try{
    const center=radioVoice.audio?.currentTime||radioTimeFromBeat();
    const clip=wavWindowBase64(radioVoice.buffer,center,20,16000);
    const request=`Listen to this actual voice/radio audio as a restrained film scorer and temporal editor.
Return exactly four labeled sections and nothing else:
TRANSCRIPT: words you can confidently hear; use [unclear] rather than inventing.
PERFORMANCE: breath, cadence, pressure, repetition, vocal grain, acceleration/deceleration, and expressive contour.
TIME: silence, phrase endings, attacks, held sounds, recurring pulses, and openings that another medium could enter.
SCORE: one concrete accompaniment move that could live beside this voice without forcing it onto a grid.
If the source contains voice plus drone/music, distinguish their roles when possible.`;

    const data=await apiFetch('/chat/completions',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        model:SKETCHRADIO_AUDIO_CHAT_MODEL,
        messages:[{role:'user',content:[
          {type:'text',text:request},
          {type:'input_audio',input_audio:{data:clip.data,format:'wav'}}
        ]}],
        max_tokens:700
      })
    });
    const raw=chatMessageText(data.choices?.[0]?.message).trim();
    if(!raw)throw new Error('Audio model returned no hearing');

    radioHearing={
      source:'raw-audio-chat',raw,
      transcript:hearingField(raw,'TRANSCRIPT'),
      performance:hearingField(raw,'PERFORMANCE'),
      time:hearingField(raw,'TIME'),
      score:hearingField(raw,'SCORE'),
      windowStart:clip.start,windowEnd:clip.end
    };
    radioVoice.reading=`RAW AUDIO ${fmtTime(clip.start)}–${fmtTime(clip.end)} · ${radioHearing.performance||raw} · ${radioHearing.time||''} · ${radioHearing.score||''}`;

    if(!$('#wordInput').value.trim()&&radioHearing.transcript){
      $('#wordInput').value=radioHearing.transcript.slice(0,900);
      fitWord();markDirty('word',true);
    }
    $('#radioReading').innerHTML=`<span class="radioHearTag">RAW AUDIO HEARD ${fmtTime(clip.start)}–${fmtTime(clip.end)}</span> · ${escapeHtml(radioHearing.performance||radioHearing.transcript||raw)}`;
    radioRenderWorld();
    await compileRadioRelations('Map three strong correspondences from what you just heard. Keep the recorded voice sovereign and preserve useful silence.',false);
    setState('RAW AUDIO HEARD · RELATIONS MAPPED','good');
  }catch(e){
    setState('AUDIO HEAR FALLBACK · READING SOUND MAP','bad');
    await deepResponsesHear();
  }finally{busy(false)}
};

// radio-deep.js bound the old function before this override loaded.
if($('#radioHear'))$('#radioHear').onclick=describeCurrentSound;
