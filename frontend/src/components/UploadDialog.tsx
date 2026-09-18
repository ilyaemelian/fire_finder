import { useState } from 'react';
import { X, Upload } from 'lucide-react';
import { upload } from '../api';
export default function UploadDialog({close,done}:{close:()=>void;done:(id:string)=>void}){
 const [busy,setBusy]=useState(false);const [error,setError]=useState('');
 return <div className="modal-backdrop"><section role="dialog" aria-modal="true" aria-labelledby="upload-title" className="modal"><button className="icon-button close" aria-label="Закрыть загрузку" disabled={busy} onClick={close}><X size={20}/></button><span className="eyebrow">РЕЗУЛЬТАТЫ МОДЕЛЕЙ</span><h2 id="upload-title">Добавить наблюдение</h2><p>Две маски и их геопривязка. После проверки результат появится в списке наборов.</p><form onSubmit={async e=>{e.preventDefault();setBusy(true);setError('');try{const id=await upload(new FormData(e.currentTarget));done(id);}catch(e){setError(String(e instanceof Error?e.message:e));}finally{setBusy(false);}}}>
 <label>Активное горение · 256 × 256 · 0/1<input name="active_fire" type="file" accept=".npy" required disabled={busy}/></label>
 <label>Степень поражения · 512 × 512 · 0–3<input name="burn_severity" type="file" accept=".npy" required disabled={busy}/></label>
 <label>Геопривязка и даты<input name="metadata" type="file" accept=".json" required disabled={busy}/></label>
 <p className="hint">Образец: data/demo-input/metadata.json. Файлы масок — uint8. Для реальных данных укажите data_mode: real.</p>
 {error&&<div role="alert" className="error">{error}</div>}<button className="primary" disabled={busy}><Upload size={16}/>{busy?'Проверяем маски…':'Загрузить результаты'}</button></form></section></div>;
}
