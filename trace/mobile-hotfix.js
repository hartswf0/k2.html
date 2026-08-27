"use strict";
/* TRACE mobile invariants: transport is never view-specific; critical actions stay thumb-reachable. */
(()=>{
  const style=document.createElement("style");
  style.id="trace-mobile-hotfix";
  style.textContent=`
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
  .transport .btn{
    display:flex!important;
    width:52px;
    min-width:52px;
    height:52px;
    min-height:52px;
    padding:0;
    font-size:18px;
  }
  .transport .btn.signal{background:#ef2200;color:#fff;border-color:#ef2200}
  .codeLine .tx{
    white-space:pre-wrap;
    overflow:visible;
    overflow-wrap:anywhere;
    word-break:break-word;
    padding-right:10px;
  }
  .codeLine{grid-template-columns:42px minmax(0,1fr)}
  .codeLine .ln{font-size:9px;padding-top:7px}
  .sourceBody,.worldBody,.sideView{padding-bottom:0}
  .trailDrawer{z-index:57}
  .onboardCard{bottom:74px}
}
@media (max-width:430px){
  .transport{left:6px;bottom:calc(62px + env(safe-area-inset-bottom))}
  .transport .btn{width:48px;min-width:48px;height:48px;min-height:48px}
}
`;
  document.head.appendChild(style);
})();
