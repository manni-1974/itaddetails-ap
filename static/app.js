(function(){
  function byId(id){ return document.getElementById(id); }
  function setStatus(msg, isErr=false){
    const s = byId('statusMsg'); if(!s) return;
    s.textContent = msg || "";
    s.className = "status " + (isErr ? "err" : "ok");
  }

  function rebuildBox(boxId, inputId){
    const box = byId(boxId);
    while(box.firstChild){ box.removeChild(box.firstChild); }
    const ph = document.createElement('span'); ph.className='ph'; ph.textContent='Tap or drop image';
    const input = document.createElement('input');
    input.className='fileInput'; input.type='file'; input.accept='image/*'; input.id = inputId;
    box.appendChild(ph); box.appendChild(input);
    input.addEventListener('change', ()=>{
      const f = input.files && input.files[0];
      const old = box.querySelector('img'); if(old) old.remove();
      const phE = box.querySelector('.ph'); if(phE) phE.remove();
      if(!f){
        if(!box.querySelector('.ph')){ const s=document.createElement('span'); s.className='ph'; s.textContent='Tap or drop image'; box.prepend(s); }
        return;
      }
      const img = document.createElement('img');
      img.src = URL.createObjectURL(f);
      img.onload = ()=> URL.revokeObjectURL(img.src);
      box.appendChild(img);
    });

    // Optional: simple drag-drop
    box.addEventListener('dragover', (e)=>{ e.preventDefault(); box.style.borderColor = '#9ab6ff'; });
    box.addEventListener('dragleave', ()=>{ box.style.borderColor = '#8ea2c9'; });
    box.addEventListener('drop', (e)=>{
      e.preventDefault(); box.style.borderColor = '#8ea2c9';
      if(e.dataTransfer.files && e.dataTransfer.files[0]){
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event('change'));
      }
    });

    return input;
  }

  function hardResetUI(){
    rebuildBox('boxA','fileA');
    rebuildBox('boxB','fileB');
    setStatus("");
    const ab = byId('analyzeBtn'); if(ab) ab.disabled = false;
  }

  function getFilesSafe(){
    let fileA = byId('fileA'); if(!fileA) fileA = rebuildBox('boxA','fileA');
    let fileB = byId('fileB'); if(!fileB) fileB = rebuildBox('boxB','fileB');
    return { fileA, fileB };
  }

  window.addEventListener('DOMContentLoaded', () => {
    hardResetUI();

    const analyzeBtn = byId('analyzeBtn');
    const resetBtn   = byId('resetBtn');
    const tbody      = byId('tbody');
    let rowCount = 0;

    resetBtn.addEventListener('click', hardResetUI);

    analyzeBtn.addEventListener('click', async () => {
      try{
        const { fileA, fileB } = getFilesSafe();
        if(!(fileA.files && fileA.files[0]) || !(fileB.files && fileB.files[0])){
          setStatus("Please add both photos before analyzing.", true);
          return;
        }
        setStatus("Analyzing…");
        analyzeBtn.disabled = true;

        const fd = new FormData();
        fd.append('fileA', fileA.files[0]);
        fd.append('fileB', fileB.files[0]);

        const r = await fetch('/analyze_pair', { method:'POST', body: fd });
        const j = await r.json().catch(()=> ({}));
        if(!r.ok){ throw new Error((j && (j.detail||j.error)) || `HTTP ${r.status}`); }

        rowCount += 1;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${rowCount}</td>
          <td>${j.manufacturer || ""}</td>
          <td>${j.model_no || j.retail_model_guess || ""}</td>
          <td>${j.serial || ""}</td>
          <td>${j.cpu || ""}</td>
          <td>${j.ram || ""}</td>
          <td><button type="button" class="btn primary">Confirm</button></td>
        `;
        const btn = tr.querySelector('button');

        btn.addEventListener('click', async () => {
          try{
            btn.disabled = true;
            const rr = await fetch('/confirm_row', {
              method:'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify(j)
            });
            const jj = await rr.json().catch(()=> ({}));
            if(!rr.ok){ btn.disabled = false; throw new Error((jj && (jj.detail||jj.error)) || `HTTP ${rr.status}`); }
            setStatus("Saved. Ready for the next two photos.");
            hardResetUI();
          }catch(e){
            setStatus("Confirm failed: " + e.message, true);
            btn.disabled = false;
          }
        });

        tbody.appendChild(tr);
        setStatus("Review then click Confirm to save.");
      }catch(e){
        setStatus("Analyze failed: " + e.message, true);
      }finally{
        analyzeBtn.disabled = false;
      }
    });
  });
})();
