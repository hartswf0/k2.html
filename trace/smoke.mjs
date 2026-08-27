import fs from "node:fs";
import path from "node:path";

const root=path.resolve(path.dirname(new URL(import.meta.url).pathname));
const read=f=>fs.readFileSync(path.join(root,f),"utf8");
const assert=(ok,msg)=>{if(!ok)throw new Error(`TRACE SMOKE · ${msg}`)};

const app=read("app.js"),index=read("index.html"),finish=read("variety-v8-finish.js"),layering=read("variety-v8-layering.js"),guard=read("variety-v8-layering-guard.js"),targetFix=read("variety-v8-target-fix.js"),mobile=read("mobile-hotfix.js"),base=read("variety-v8-base.js"),flow=read("variety-v8-flow.js"),fixes=read("variety-v8-fixes.js"),runtime=read("variety-v8-runtime.js");

assert(/const version="8\.7"/.test(app),"app is not v8.7");
assert(/app\.js\?v=8\.7/.test(index),"index cache-bust is not v8.7");
assert(!app.includes('"variety-v8.js"'),"unsafe legacy v8 wrapper is in boot chain");

const listBlock=app.match(/const files=\[([\s\S]*?)\];/);assert(listBlock,"boot file list missing");
const files=[...listBlock[1].matchAll(/"([^"]+\.js)"/g)].map(m=>m[1]);
assert(files.includes("variety-v8-finish.js"),"release gate not in boot chain");
assert(files.includes("variety-v8-layering.js"),"low-road layering module not in boot chain");
assert(files.includes("variety-v8-layering-guard.js"),"focused-track guard not in boot chain");
assert(files.includes("variety-v8-target-fix.js"),"hard target compiler not in boot chain");
assert(files.indexOf("variety-v8-finish.js")<files.indexOf("variety-v8-layering.js"),"layering must inherit release bounds");
assert(files.indexOf("variety-v8-layering.js")<files.indexOf("variety-v8-layering-guard.js"),"target guard must follow layering");
assert(files.indexOf("variety-v8-layering-guard.js")<files.indexOf("variety-v8-target-fix.js"),"hard target compiler must follow focused voice guard");
assert(files.indexOf("variety-v8-target-fix.js")<files.indexOf("variety-v8-boot.js"),"hard target compiler must load before v8 boot");
for(const f of files)assert(fs.existsSync(path.join(root,f)),`boot dependency missing: ${f}`);

for(const id of ["playBtn","stopBtn","sendBtn","trailBtn","timeline","messages","prompt","sourceEdit"])assert(index.includes(`id="${id}"`),`critical surface missing: ${id}`);
const ids=[...index.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
const dup=ids.filter((x,i)=>ids.indexOf(x)!==i);assert(!dup.length,`duplicate DOM ids: ${[...new Set(dup)].join(", ")}`);

assert(/position:fixed/.test(mobile)&&/\.transport/.test(mobile),"mobile transport is not fixed/reachable");
assert(/\.trailDrawer/.test(mobile)&&/overflow-x:auto/.test(mobile),"mobile run trail is not horizontally reachable");
assert(/TRACE_V7_BASE=Object\.freeze/.test(base),"explicit v7 compatibility seam missing");
assert(/variants=\[root,v8PlanVariant\(text,root,1\),v8PlanVariant\(text,root,2\)\]/.test(flow),"three candidate paths missing");
assert(/\["ROOT","CROSS","WILD"\]/.test(fixes),"branch labels missing");
assert(/instrument\(\$\{lastTrack\}/.test(finish),"executable instrument patch missing");
assert(/boundRepeats/.test(finish)&&/boundPlan/.test(finish),"scope-bound release gate missing");
assert(/TRACE_RELEASE_SELF_TEST/.test(finish),"browser startup self-test missing");
assert(/V8_GM_NAMES/.test(runtime)&&/V8_GM_SLUGS/.test(runtime),"GM repertoire missing");

assert(/TRACE_LAYER/.test(layering),"prompt target state missing");
assert(/mode:\"track\"/.test(layering)&&/mode:\"layer\"/.test(layering)&&/mode:\"all\"/.test(layering),"ALL/TRACK/NEW PART target modes missing");
assert(/p\.clear_ranges=\[\]/.test(layering),"NEW PART is not explicitly additive");
assert(/replace only CH/.test(layering),"focused-track replacement bound missing");
assert(/prompt_target/.test(layering)&&/trace_target/.test(layering),"target provenance missing");
assert(/traceTargetNew/.test(layering)&&/\+ PART/.test(layering),"mobile-visible new part control missing");
assert(/normalizeFocusedVoice/.test(guard),"focused voice normalization missing");
assert(/HARD TARGET/.test(targetFix),"hard target explanation missing");
assert(/const percussion=\[\"drum\",\"percussion\"\]/.test(layering),"legacy carrier grammar seam changed unexpectedly");
assert(/const scope=planScope\(p\).*rk=requestedKind\(text\).*kind=rk\|\|/.test(targetFix.replace(/\n/g," ")),"requested carrier kind does not drive target grammar");
assert(/conversion\|\|procedural\|\|!p\.note_loops\.length/.test(targetFix),"melodic carrier conversion fallback missing");
assert(/conversion\|\|explicit!==null\)p\.sample_assignments=\[\]/.test(targetFix),"sample identity is not cleared on explicit carrier conversion");
assert(/SEND CH/.test(targetFix)&&/ADD PART/.test(targetFix),"target-specific send affordance missing");

let tagCount=0;
for(let i=1;i<=8;i++){
  const f=path.join(root,"tags",`treblo-${String(i).padStart(2,"0")}.txt`);
  assert(fs.existsSync(f),`tag shard missing: ${path.basename(f)}`);
  tagCount+=fs.readFileSync(f,"utf8").split(/\r?\n/).filter(Boolean).length;
}
assert(tagCount===4160,`expected 4160 repertoire tags, found ${tagCount}`);

console.log(`TRACE SMOKE OK · ${files.length} boot scripts · hard prompt targets · ${tagCount} tags · ${new Set(ids).size} unique DOM ids`);
