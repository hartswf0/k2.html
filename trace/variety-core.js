"use strict";
/* TRACE v7 variety layer.
   Explicit compatibility layer: expands orchestration, synthesis, MIDI, and visible history
   without invalidating v6 sessions or the compact TRACE source language. */

const V7_TRACK_LIMIT=16;
const V7_EVENT_LIMIT=960;
const V7_NOTE_LOOP_LIMIT=16;
const V7_DRUM_LOOP_LIMIT=16;
const V7_EFFECT_LIMIT=16;
const V7_DRUMS=new Set(["kick","snare","hat","noise","knock","mallet","clap","tom","rim","shaker","cowbell","conga","clave","woodblock","tambourine","bell"]);
const V7_KINDS=new Set(["synth","lead","bass","drum","electric-piano","organ","guitar","horn","reed","strings","flute","mallet","percussion","texture","voice"]);
const V7_TRACK_PALETTE=Object.freeze([
  [1,"LEAD","lead","sawtooth"],
  [2,"BASS","bass","triangle"],
  [3,"KIT","drum","noise"],
  [4,"KEYS","electric-piano","sine"],
  [5,"ORGAN","organ","sine"],
  [6,"GUITAR","guitar","triangle"],
  [7,"HORNS","horn","sawtooth"],
  [8,"REEDS","reed","square"],
  [9,"STRINGS","strings","sawtooth"],
  [10,"MALLETS","mallet","sine"],
  [11,"PERC","percussion","noise"],
  [12,"TEXTURE","texture","sine"]
]);
const V7_DISTANT_STONES=Object.freeze([
  "gnawa","ethio-jazz","highlife","mbalax","gamelan","qawwali","joik","tuvan throat singing",
  "musique concrete","acousmatic","spectralism","microtonal","prepared piano","free improvisation",
  "dub poetry","no wave","footwork","juke","broken beat","plunderphonics","field recording","sound collage",
  "kora","oud","marimba","vibraphone","hurdy gurdy","steel band","afro-cuban jazz","spiritual jazz",
  "krautrock","library music","minimal synth","drumless","ritual ambient","turntablism","chopped and screwed"
]);
const V7_COMMON_TAGS=`psychedelic soul|southern soul|afrobeat|afro-jazz|spiritual jazz|free jazz|modal jazz|hard bop|ethio-jazz|gnawa|highlife|mbalax|soukous|gamelan|qawwali|joik|tuvan throat singing|kora|oud|marimba|vibraphone|prepared piano|field recording|sound collage|musique concrete|acousmatic|spectralism|microtonal|plunderphonics|turntablism|dub poetry|no wave|footwork|juke|broken beat|drumless|krautrock|minimal synth|ritual ambient|ambient dub|dub techno|acid jazz|jazz-funk|jazz fusion|avant-garde jazz|country soul|country rock|appalachian folk|nashville sound|honky tonk|outlaw country|bluegrass|western swing|psychedelic folk|folk rock|jam band|roots rock|funk|p-funk|deep funk|motown|neo-soul|gospel|rhythm and blues|quiet storm|boogie|disco|house|acid house|deep house|techno|detroit techno|jungle|drum and bass|breakbeat|breakcore|garage|uk garage|grime|drill|boom bap|g-funk|cloud rap|trap|phonk|reggaeton|dancehall|roots reggae|ska|rocksteady|cumbia|salsa|samba|bossa nova|tango|flamenco|fado|bolero|carnatic|hindustani|arabic classical|persian classical|klezmer|balkan brass band|celtic folk|afro-cuban jazz|latin jazz|chiptune|vaporwave|hyperpop|industrial|noise|harsh noise|electroacoustic|minimalism|serialism|twelve-tone|aleatory|free improvisation|polyrhythm|uncommon time signatures`.split("|");
let v7Tags=[];
let v7TagPromise=null;
let v7LastTagCandidates=[];

function v7Track(id,name,kind,wave){return{id:+id,name:String(name||`CH ${id}`).slice(0,24),kind:V7_KINDS.has(kind)?kind:"synth",wave:WAVES.has(wave)?wave:"sine"}}
function track(id,name,kind,wave){return v7Track(id,name,kind,wave)}
function drum(trackId,kind,start,duration,velocity,prov={}){return{id:nextEventId++,type:"drum",track:+trackId,drum:V7_DRUMS.has(kind)?kind:"noise",start:+start,duration:+duration,velocity:clamp(Math.round(velocity),1,127),provenance:{origin:prov.origin||"source",reason:prov.reason||"",promptId:prov.promptId||null,opIndex:Number.isInteger(prov.opIndex)?prov.opIndex:null,sourceLine:null}}}
function ensure(c){if(!c||!Number.isFinite(c.bpm)||c.bpm<30||c.bpm>280)throw Error("Tempo must be 30–280 BPM.");if(!Number.isInteger(c.bars)||c.bars<1||c.bars>64)throw Error("Bars must be 1–64.");if(!Array.isArray(c.tracks)||c.tracks.length<1||c.tracks.length>V7_TRACK_LIMIT)throw Error(`Track limit is ${V7_TRACK_LIMIT}.`);if(!Array.isArray(c.events)||c.events.length>V7_EVENT_LIMIT)throw Error(`Event limit is ${V7_EVENT_LIMIT}.`);const ids=new Set();for(const t of c.tracks){if(!Number.isInteger(+t.id)||ids.has(+t.id))throw Error("Track ids must be unique integers.");ids.add(+t.id)}for(const e of c.events){if(!ids.has(+e.track))throw Error("Event references missing track.");if(!Number.isFinite(e.start)||!Number.isFinite(e.duration)||e.start<0||e.duration<=0||e.start>=beats(c)+.001)throw Error("Invalid event timing.")}return c}
function seed(){const c={version:7,title:"FIRST TRACE",bpm:112,bars:8,beatsPerBar:4,tracks:V7_TRACK_PALETTE.map(x=>v7Track(...x)),events:[],fx:{}};for(const t of c.tracks)c.fx[String(t.id)]=fx0();const prov=reason=>({reason,origin:"source"});
  [0,2,4,6,8,10,12,14].forEach((s,i)=>c.events.push(note(2,[43,43,46,48][i%4],s,.72,84,"triangle",prov("sparse bass foothold"))));
  [0,4,8,12].forEach(s=>c.events.push(drum(3,"kick",s,.12,108,prov("four-bar foothold"))));
  [2,6,10,14].forEach(s=>c.events.push(drum(3,"snare",s,.12,92,prov("backbeat foothold"))));
  [1.5,3.5,5.5,7.5,9.5,11.5,13.5,15.5].forEach(s=>c.events.push(drum(11,"shaker",s,.1,45,prov("loose peripheral pulse"))));
  [0,4,8,12].forEach((s,i)=>c.events.push(note(4,[62,65,67,60][i],s,1.8,62,"sine",prov("quiet harmonic marker"))));
  return ensure(c)}
function v7EnsurePalette(c){if(!c||!Array.isArray(c.tracks))return c;const ids=new Set(c.tracks.map(t=>+t.id));for(const spec of V7_TRACK_PALETTE){if(c.tracks.length>=12)break;if(!ids.has(spec[0])){c.tracks.push(v7Track(...spec));ids.add(spec[0])}}for(const t of c.tracks)c.fx[String(t.id)]||=fx0();c.version=Math.max(7,+c.version||0);return ensure(c)}

function parseSource(text){const c={version:7,title:state.composition.title,bpm:120,bars:8,beatsPerBar:4,tracks:[],events:[],fx:{}},lines=text.split(/\r?\n/);if(lines.length>1200)throw Error("Source too long.");for(let li=0;li<lines.length;li++){const raw=lines[li].trim();if(!raw||raw.startsWith("#")||raw.startsWith("//"))continue;let m;if((m=raw.match(/^tempo\(([-\d.]+)\)$/))){c.bpm=+m[1];continue}if((m=raw.match(/^bars\((\d+)\)$/))){c.bars=+m[1];continue}if((m=raw.match(/^track\((\d+),\s*("(?:[^"\\]|\\.)*"),\s*("(?:[^"\\]|\\.)*"),\s*("(?:[^"\\]|\\.)*")\)$/))){const id=+m[1],name=JSON.parse(m[2]),kind=JSON.parse(m[3]),wave=JSON.parse(m[4]);c.tracks.push(v7Track(id,name,kind,wave));c.fx[String(id)]=fx0();continue}if((m=raw.match(/^loopNote\((\d+),\s*(\[[^\]]*\]),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*(\d+),\s*(\d+),\s*("(?:[^"\\]|\\.)*")\)$/))){const tid=+m[1],p=JSON.parse(m[2]),start=+m[3],step=+m[4],dur=+m[5],rep=+m[6],vel=+m[7],wave=JSON.parse(m[8]);if(!Array.isArray(p)||p.length<1||p.length>32||rep<1||rep>96)throw Error(`Line ${li+1}: loop bound invalid.`);for(let i=0;i<rep;i++){c.events.push(note(tid,+p[i%p.length],start+i*step,dur,vel,wave,{reason:`source line ${li+1}`}));if(c.events.length>V7_EVENT_LIMIT)throw Error(`Line ${li+1}: event limit exceeded.`)}continue}if((m=raw.match(/^loopDrum\((\d+),\s*("(?:[^"\\]|\\.)*"),\s*([-\d.]+),\s*([-\d.]+),\s*(\d+),\s*(\d+)\)$/))){const tid=+m[1],kind=JSON.parse(m[2]),start=+m[3],step=+m[4],rep=+m[5],vel=+m[6];if(rep<1||rep>96)throw Error(`Line ${li+1}: loop bound invalid.`);for(let i=0;i<rep;i++){c.events.push(drum(tid,kind,start+i*step,.12,vel,{reason:`source line ${li+1}`}));if(c.events.length>V7_EVENT_LIMIT)throw Error(`Line ${li+1}: event limit exceeded.`)}continue}if((m=raw.match(/^fx\((\d+),\s*("(?:[^"\\]|\\.)*"),\s*([-\d.]+)\)$/))){const tid=String(+m[1]),p=JSON.parse(m[2]),v=+m[3];if(!FX.has(p))throw Error(`Line ${li+1}: unknown effect.`);c.fx[tid]||=fx0();c.fx[tid][p]=v;continue}throw Error(`Line ${li+1}: unsupported TRACE statement.`)}return ensure(c)}