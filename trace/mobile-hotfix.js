"use strict";
/* Mobile operational invariants: transport, history, and code remain reachable in every surface. */
(()=>{
  const style=document.createElement("style");
  style.id="trace-mobile-hotfix";
  style.textContent=`
.patchDetails{margin-top:6px;border-top:1px solid #000}.patchDetails summary{list-style:none;cursor:pointer;padding:8px 0;font:900 8px/1 Arial;letter-spacing:.06em}.patchDetails summary::-webkit-details-marker{display:none}.patchDetails[open] summary{border-bottom:1px solid #000}.patchDetails .sourcePatch{max-height:42dvh;overflow:auto;margin:0;border:0;padding:8px 0}
.runNode{display:flex;flex-direction:column;align-items:flex-start;justify-content:flex-start;gap:5px;text-align:left}.runNode span{font:800 8px/1.15 Arial}.runNode small{font:600 8px/1.25 Arial;opacity:.72}.stone{display:flex;flex-direction:column;gap:5px}.stone span{font:700 8px/1.25 Arial}.stone code{display:block;white-space:pre-wrap;overflow-wrap:anywhere;font:600 8px/1.3 ui-monospace,monospace}
@media (max-width:820px){
  .transport{
    display:flex!important;
    position:fixed;
    left:8px;
    bottom:calc(64px + env(safe-area-inset-bottom));
    z-index:58;
    gap:4px;
    padding:4px;
    background:#fff;
    border:2px solid #000;
    box-shadow:4px 4px 0 #000;
  }
  .transport .btn{display:flex!important;align-items:center;justify-content:center;width:52px;min-width:52px;height:52px;min-height:52px;padding:0;font-size:18px}
  .transport .btn.signal{background:#ef2200;color:#fff;border-color:#ef2200}
  .codeLine .tx{white-space:pre-wrap;overflow:visible;overflow-wrap:anywhere;word-break:break-word;padding-right:10px}
  .codeLine{grid-template-columns:42px minmax(0,1fr)}
  .codeLine .ln{font-size:9px;padding-top:7px}
  .sourceBody,.worldBody,.sideView{padding-bottom:0}
  .onboardCard{bottom:74px}

  /* TRAIL is a real mobile surface, not a short drawer with clipped runs. */
  .trailDrawer{top:calc(38px + env(safe-area-inset-top));bottom:calc(56px + env(safe-area-inset-bottom));height:auto;z-index:57;box-shadow:none;border-top:2px solid #000;background:#fff;overflow:hidden}
  .trailHead{height:48px;min-height:48px;padding:0 10px;font-size:9px}.trailHead span:not(.spacer){display:none}.trailHead .micro{min-height:38px}
  .trailBody{height:calc(100% - 48px);display:grid;grid-template-columns:1fr;grid-template-rows:minmax(138px,38%) minmax(0,62%);overflow:hidden}
  .trailGroup{height:auto!important;min-height:0;padding:10px;border-right:0;border-bottom:2px solid #000;overflow:hidden;display:flex;flex-direction:column}
  .trailGroup:last-child{border-bottom:0}.trailGroup>b{font-size:10px;margin-bottom:8px}
  .path{display:flex!important;gap:8px!important;height:auto!important;min-height:0;flex:1;overflow-x:auto!important;overflow-y:hidden!important;padding:0 0 7px;scroll-snap-type:x proximity;-webkit-overflow-scrolling:touch}
  .runNode{flex:0 0 154px!important;width:154px!important;min-width:154px!important;height:94px!important;min-height:94px!important;padding:10px!important;border:1.5px solid #000!important;background:#fff!important;color:#000!important;scroll-snap-align:start;overflow:hidden}
  .runNode.current{background:#000!important;color:#fff!important;outline:3px solid #ef2200;outline-offset:-3px}.runNode b{font-size:11px}.runNode small{display:block;max-height:32px;overflow:hidden}
  .stones{display:grid!important;grid-auto-flow:column;grid-auto-columns:min(78vw,280px);grid-template-rows:1fr;height:auto!important;min-height:0;flex:1;gap:8px!important;overflow-x:auto!important;overflow-y:hidden!important;padding:0 0 7px;-webkit-overflow-scrolling:touch}
  .stone{height:100%!important;min-width:0!important;padding:10px!important;border:1px solid #000!important;overflow:hidden}.stone code{max-height:94px;overflow:hidden}.stone .btn{margin-top:auto;align-self:flex-start}
  #trailBtn.active{background:#000;color:#fff}

  .messages{padding-bottom:76px}.composer{padding-bottom:max(76px,env(safe-area-inset-bottom))}
  .msg{font-size:10px;line-height:1.35}.planRow{grid-template-columns:54px 1fr;font-size:9px}.msgActions{position:sticky;bottom:0;background:#fff;padding:5px 0;z-index:2}
  .timelineWrap{overscroll-behavior:contain}.track{height:60px}
  #timeline{min-width:900px}
}
@media (max-width:430px){
  .transport{left:6px;bottom:calc(62px + env(safe-area-inset-bottom))}
  .transport .btn{width:48px;min-width:48px;height:48px;min-height:48px}
  .runNode{flex-basis:144px!important;width:144px!important;min-width:144px!important}
}
`;
  document.head.appendChild(style);
})();
