import { useState } from 'react';
import { X, Upload } from 'lucide-react';
import { upload, uploadSubmission } from '../api';

export default function UploadDialog({close,done}:{close:()=>void;done:(id:string)=>void}){
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[mode,setMode]=useState<'submission'|'masks'>('submission');
 return <div className="modal-backdrop"><section role="dialog" aria-modal="true" aria-labelledby="upload-title" className="modal">
  <button className="icon-button close" aria-label="Закрыть загрузку" disabled={busy} onClick={close}><X size={20}/></button>
  <span className="eyebrow">РЕЗУЛЬТАТЫ МОДЕЛЕЙ</span><h2 id="upload-title">Добавить наблюдения</h2>
  <div className="upload-modes">
   <button type="button" className={mode==='submission'?'active':''} onClick={()=>{setMode('submission');setError('');}}>Итоговый CSV</button>
   <button type="button" className={mode==='masks'?'active':''} onClick={()=>{setMode('masks');setError('');}}>Две маски</button>
  </div>
  <form key={mode} onSubmit={async e=>{e.preventDefault();setBusy(true);setError('');try{const data=new FormData(e.currentTarget);const id=mode==='submission'?await uploadSubmission(data):await upload(data);done(id);}catch(e){setError(String(e instanceof Error?e.message:e));}finally{setBusy(false);}}}>
   {mode==='submission'?<>
    <p>Конкурсный CSV загружается без изменения структуры. Координаты и даты передаются отдельным JSON.</p>
    <label>submission.csv · chip_id, class_id, rle<input name="submission" type="file" accept=".csv,text/csv" required disabled={busy}/></label>
    <label>Геопривязка чипов и даты<input name="metadata" type="file" accept=".json,application/json" required disabled={busy}/></label>
    <p className="hint">Формат JSON приведён в examples/submission_metadata.example.json. Все chip_id из CSV должны иметь запись в JSON.</p>
   </>:<>
    <p>Режим для одной пары выходных матриц AF и BS.</p>
    <label>Активное горение · 256 × 256 · 0/1<input name="active_fire" type="file" accept=".npy" required disabled={busy}/></label>
    <label>Степень поражения · 512 × 512 · 0–3<input name="burn_severity" type="file" accept=".npy" required disabled={busy}/></label>
    <label>Геопривязка и даты<input name="metadata" type="file" accept=".json" required disabled={busy}/></label>
    <p className="hint">Образец создаётся в data/demo-input/metadata.json командой запуска сервиса.</p>
   </>}
   {error&&<div role="alert" className="error">{error}</div>}
   <button className="primary" disabled={busy}><Upload size={16}/>{busy?'Проверяем данные…':'Загрузить результаты'}</button>
  </form>
 </section></div>;
}
