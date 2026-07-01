/** Minimal self-contained chat demo UI served at GET /. Vanilla JS, streams via SSE. */
export const INDEX_HTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neuromod Chat</title><style>
 body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0}
 .wrap{max-width:720px;margin:0 auto;padding:16px}
 h1{font-size:18px}
 .ctl{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
 select,input,button,textarea{background:#1b1e27;color:#e6e6e6;border:1px solid #333;border-radius:6px;padding:6px}
 #log{white-space:pre-wrap;background:#141721;border:1px solid #2a2f3a;border-radius:8px;padding:12px;min-height:200px}
 label{font-size:13px;color:#9aa4b2} .dose{color:#e67e22}
 button{cursor:pointer;background:#c0392b;border-color:#c0392b}
</style></head><body><div class="wrap">
<h1>Neuromod Chat <span style="color:#666">— dosed inference</span></h1>
<div class="ctl">
 <label>Pack <select id="pack"></select></label>
 <label>Intensity <span class="dose" id="dv">0.50</span>
  <input type="range" id="intensity" min="0" max="1" step="0.05" value="0.5"></label>
</div>
<textarea id="prompt" rows="2" style="width:100%" placeholder="Say something…">Describe a sunset.</textarea>
<div class="ctl"><button id="send">Send (stream)</button></div>
<div id="log"></div>
</div><script>
const $=id=>document.getElementById(id);
$('intensity').oninput=e=>$('dv').textContent=(+e.target.value).toFixed(2);
fetch('/api/packs').then(r=>r.json()).then(d=>{
  const sel=$('pack'); (d.packs||['none','lsd','dmt','cocaine','mdma']).forEach(p=>{
    const o=document.createElement('option');o.value=o.textContent=p;sel.appendChild(o);});
}).catch(()=>{});
$('send').onclick=async()=>{
  const log=$('log');log.textContent='';
  const body={messages:[{role:'user',content:$('prompt').value}],
    pack_name:$('pack').value==='none'?null:$('pack').value,intensity:+$('intensity').value};
  const res=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  const reader=res.body.getReader();const dec=new TextDecoder();let buf='';
  while(true){const{done,value}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});
    let i;while((i=buf.indexOf('\\n\\n'))>=0){const frame=buf.slice(0,i);buf=buf.slice(i+2);
      const line=frame.replace(/^data: /,'');if(line==='[DONE]')continue;
      try{const o=JSON.parse(line);if(o.chunk)log.textContent+=o.chunk;}catch(e){}}}
};
</script></body></html>`;
