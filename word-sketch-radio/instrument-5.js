'use strict';
(()=>{
  const base=document.createElement('script');
  base.src='../word-sketch-song/instrument-5.js';
  base.onload=()=>{
    const radio=document.createElement('script');
    radio.src='radio.js';
    document.body.appendChild(radio);
  };
  document.body.appendChild(base);
})();