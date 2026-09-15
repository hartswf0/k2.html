const fs=require('node:fs');
(async function testController(coreCode,appCode){
const C=new Function(coreCode+'return globalThis.PondCore;')();
class Element{constructor(){this.value='';this.textContent='';this.children=[];this.dataset={};this.style={setProperty(){}};this.classList={add(){},remove(){},toggle(){}}}append(...e){this.children.push(...e)}setAttribute(){}showModal(){}close(){}focus(){}click(){return this.onclick?.({})}set innerHTML(x){this.children=[]}}
class Abort{constructor(){this.signal={aborted:false}}abort(){this.signal.aborted=true}}
const els=new Map(),doc={getElementById(id){if(!els.has(id))els.set(id,new Element());return els.get(id)},createElement(){return new Element()},querySelectorAll(){return []}};
const storage=()=>{const m=new Map();return{getItem:k=>m.get(k)||null,setItem:(k,v)=>m.set(k,v),removeItem:k=>m.delete(k)}};
const tab=storage();tab.setItem('pond.key','test-only');const root={PondCore:C},$=id=>doc.getElementById(id);$('scope').value='both';let bad=false;
const fakeFetch=async(url,opt)=>{const {context}=JSON.parse(JSON.parse(opt.body).input),p=C.clone(context.passage);p.summary='Test candidate';if(bad)p.tracks[0].bars[0][0].velocity=.01;else p.lyrics[0]='Generated line';return{ok:true,json:async()=>({output_text:JSON.stringify(p)})}};
new Function('globalThis','document','window','localStorage','sessionStorage','fetch','clearInterval','AbortController',appCode)(root,doc,{},storage(),tab,fakeFetch,()=>{},Abort);
const checks=[],ok=(x,n)=>{if(!x)throw Error(n);checks.push(n)};
ok($('timeline').children.length===6,'Timeline renders ruler, four tracks and lyrics');
$('variation').click();ok(!!root.Pond.session.snapshot().candidate,'Variation creates candidate');$('commit').click();ok(root.Pond.session.snapshot().revision===1,'Commit advances revision');$('undo').click();ok(C.equal(root.Pond.session.snapshot().composition,C.starter()),'Undo restores original');
$('lyrics').value='Draft line';$('lyrics').oninput();$('start').value='5';$('start').onchange();$('start').value='1';$('start').onchange();$('end').value='4';$('end').onchange();ok($('lyrics').value==='Draft line','Writing draft survives selection changes');
$('applyLyrics').click();ok(root.Pond.session.snapshot().composition.lyrics[0].text==='Draft line','Apply lyrics writes passage');$('direction').value='New words';await $('generate').click();ok(root.Pond.session.snapshot().candidate.proposal.lyrics[0]==='Generated line','Mock API result becomes candidate');ok(root.Pond.session.snapshot().composition.lyrics[0].text==='Draft line','Generation preserves committed lyrics');
root.Pond.session.mutate('lock',c=>c.tracks[0].locked=true);bad=true;await $('generate').click();ok(!root.Pond.session.snapshot().candidate,'Invalid generated lock change rejected');ok($('status').textContent.includes('protected'),'Rejection visible in status');
return checks;
})(fs.readFileSync(__dirname+'/pond-core.js','utf8'),fs.readFileSync(__dirname+'/pond-app.js','utf8')).then(r=>console.log(r.join('\n'))).catch(e=>{console.error(e);process.exitCode=1});
