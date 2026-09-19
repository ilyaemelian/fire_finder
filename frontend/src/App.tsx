import { useEffect, useRef, useState } from 'react';
import { Flame, Layers, Upload, ArrowDownToLine, Scan, X, ArrowRight, MapPin, Info, Activity } from 'lucide-react';
import type { Polygon } from 'geojson';
import FireMap, { colors } from './components/FireMap';
import UploadDialog from './components/UploadDialog';
import * as api from './api';
import { polygon, type Dataset, type Result } from './types';
const number=(value:number)=>value.toLocaleString('ru-RU',{maximumFractionDigits:2});
const DAY=86_400_000;
const day=(value:string)=>Math.floor(Date.parse(value+'T00:00:00Z')/DAY);
const date=(value:number)=>new Date(value*DAY).toISOString().slice(0,10);
function TimeRange({dataset,from,to,onChange}:{dataset:Dataset;from:string;to:string;onChange:(from:string,to:string)=>void}){
 const min=day(dataset.date_from),max=day(dataset.date_to),span=Math.max(1,max-min);
 const left=Math.max(0,Math.min(span,day(from)-min)),right=Math.max(0,Math.min(span,day(to)-min));
 return <div className="time-filter"><div className="time-values"><span>{from}</span><span>{to}</span></div><div className="range-pair">
  <input aria-label="Начало периода" type="range" min={0} max={span} value={left} disabled={max===min} onChange={e=>{const value=Math.min(Number(e.target.value),right);onChange(date(min+value),to);}}/>
  <input aria-label="Конец периода" type="range" min={0} max={span} value={right} disabled={max===min} onChange={e=>{const value=Math.max(Number(e.target.value),left);onChange(from,date(min+value));}}/>
 </div></div>;
}
export default function App(){
 const [datasets,setDatasets]=useState<Dataset[]>([]),[selected,setSelected]=useState('');
 const [from,setFrom]=useState(''),[to,setTo]=useState('');
 const [selection,setSelection]=useState<Polygon|null>(null),[drawing,setDrawing]=useState(false);
 const [result,setResult]=useState<Result|null>(null),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const [showAF,setAF]=useState(true),[showBS,setBS]=useState(true),[upload,setUpload]=useState(false),[loaded,setLoaded]=useState(false);
 const [report,setReport]=useState(false);const request=useRef(0);
 const dataset=datasets.find(d=>d.id===selected);
 const invalidate=()=>{request.current++;setResult(null);setReport(false);setBusy(false);setError('');};
 function choose(d:Dataset){invalidate();setSelected(d.id);setFrom(d.date_from);setTo(d.date_to);setSelection(null);setDrawing(false);}
 async function load(id?:string){try{const items=await api.catalog();setDatasets(items);const d=items.find(x=>x.id===id)||items[0];if(d)choose(d);setLoaded(true);}catch(e){setError('Не удалось подключиться к API. Проверьте, что сервер запущен.');setLoaded(true);}}
 useEffect(()=>{void load();return ()=>{request.current++;};},[]);
 async function run(region:Polygon,asReport:boolean){if(!dataset)return;const serial=++request.current;setBusy(true);setError('');setResult(null);setReport(false);try{const res=await api.analyze({dataset_id:dataset.id,geometry:region,date_from:from,date_to:to});if(serial===request.current){setResult(res);setReport(asReport);}}catch(e){if(serial===request.current)setError(e instanceof Error?e.message:String(e));}finally{if(serial===request.current)setBusy(false);}}
 useEffect(()=>{if(dataset&&from&&to&&from<=to)void run(polygon(dataset.bounds),false);},[selected,from,to]);
 useEffect(()=>{const fn=(e:KeyboardEvent)=>{if(e.key==='Escape')setDrawing(false);};window.addEventListener('keydown',fn);return()=>window.removeEventListener('keydown',fn);},[]);
 const summary=report?result?.summary:null;
 const noCoverage=result&&!result.coverage.af.has_observations&&!result.coverage.bs.has_observations;
 return <div className="app"><header className="topbar"><div className="brand"><span className="brand-icon"><Flame size={24}/></span><div><strong>КОНТУР<span className="brand-dot">.</span></strong><small>Мониторинг природных пожаров</small></div></div><nav><span className="nav-active">Карта наблюдений</span><a href="/docs" target="_blank" rel="noreferrer">REST API ↗</a></nav><button className="secondary upload-top" onClick={()=>setUpload(true)}><Upload size={15}/>Добавить результаты</button></header>
 <main className="workspace"><aside className="sidebar"><div className="section-heading"><span className="eyebrow">ОБЗОР ТЕРРИТОРИИ</span><span className="tag">MVP / 01</span></div><h1>Пожары и гари</h1><p className="intro">Наблюдения со спутников.<br/>От обнаружения — к оценке площади.</p>
 <section className="filters"><label>Набор наблюдений<select value={selected} disabled={!datasets.length} onChange={e=>{const d=datasets.find(x=>x.id===e.target.value);if(d)choose(d);}}>{datasets.length?datasets.map(d=><option key={d.id} value={d.id}>{d.title}</option>):<option>Нет загруженных результатов</option>}</select></label>{dataset&&<TimeRange dataset={dataset} from={from} to={to} onChange={(start,end)=>{invalidate();setFrom(start);setTo(end);}}/>}<p className="hint">Интервал наблюдений, UTC</p></section>
 <section className="layer-section"><h2><Layers size={16}/>Слои карты</h2><label className="layer"><span className="layer-symbol fire"><Flame size={16}/></span><span>Активное горение<small>Обнаружения в момент съёмки</small></span><input type="checkbox" checked={showAF} onChange={e=>setAF(e.target.checked)}/></label><label className="layer"><span className="layer-symbol burn"><Layers size={16}/></span><span>Пройденная огнём площадь<small>Три степени поражения</small></span><input type="checkbox" checked={showBS} onChange={e=>setBS(e.target.checked)}/></label></section>
 <section className="analysis-section"><div className="section-heading"><h2>Справка по области</h2><Scan size={17}/></div><p>Выделите участок на карте, чтобы рассчитать площадь и скачать результат.</p><button className={'select-button '+(drawing?'active':'')} disabled={!dataset} onClick={()=>setDrawing(!drawing)}><Scan size={17}/>{drawing?'Отменить выделение':selection?'Выделить другую область':'Выделить область'}<ArrowRight size={16}/></button>{selection&&<div className="selection-note"><span><MapPin size={13}/>Область выбрана</span><button onClick={()=>{invalidate();setSelection(null);if(dataset)void run(polygon(dataset.bounds),false);}} aria-label="Сбросить область"><X size={15}/></button></div>}<button className="primary" disabled={!selection||busy||!from||!to||from>to} onClick={()=>selection&&void run(selection,true)}>{busy?'Выполняем запрос…':'Рассчитать по области'}</button>
 {summary?<div className="summary"><span className="eyebrow">ГАРЬ В ВЫБРАННОЙ ОБЛАСТИ</span><div className="total">{number(summary.total_burned_area_ha)} <span>га</span></div><div className="severity-bar">{[1,2,3].map(k=><i key={k} style={{background:colors[k],flex:summary.area_by_severity_ha[k]||0}}/>)}</div>{[1,2,3].map(k=><div className="severity-row" key={k}><span><i style={{background:colors[k]}}/>{['','Слабая','Средняя','Сильная'][k]}</span><strong>{number(summary.area_by_severity_ha[k])} га</strong></div>)}<div className="summary-foot">{summary.active_detection_count} обнаружений горения · {summary.burned_contour_count} контуров</div><div className="exports">{(['contours','summary'] as const).map(kind=><button key={kind} className="secondary" onClick={()=>result&&api.download(result,kind).catch(e=>setError(e.message))}><ArrowDownToLine size={14}/>{kind==='contours'?'GeoJSON':'Справка JSON'}</button>)}</div></div>:<div className="empty-summary"><Scan size={21}/><span>Площадь появится после расчёта<br/>по выбранной области</span></div>}
 </section><footer className="sidebar-footer"><Activity size={14}/><span>КосмоХакатон 2026 · Командный прототип</span></footer></aside>
 <section className="map-panel" aria-label="Карта наблюдений">{dataset?<FireMap dataset={dataset} result={result} selection={selection} drawing={drawing} onSelect={p=>{request.current++;setBusy(false);setError('');setReport(false);setSelection(p);setDrawing(false);}} showAF={showAF} showBS={showBS}/>:<div className="map-empty"><MapPin size={36}/><h2>{loaded?'Наблюдений пока нет':'Загружаем карту…'}</h2><p>Добавьте маски и метаданные или создайте учебный набор.</p><button className="secondary" onClick={()=>setUpload(true)}>Добавить результаты</button></div>}
 {dataset&&<div className={'map-notice '+(dataset.data_mode==='synthetic'?'synthetic':'')}><Info size={16}/><div><strong>{dataset.data_mode==='synthetic'?'Демонстрационные данные':'Результаты моделей'}</strong><span>{dataset.data_mode==='synthetic'?'Синтетические маски. Не реальные пожары.':dataset.source}</span></div><span className="notice-date">{from} — {to}</span></div>}
 {error&&<div role="alert" className="map-error">{error}<button onClick={()=>setError('')} aria-label="Закрыть ошибку"><X size={16}/></button></div>}
 {busy&&<div className="map-status">Обрабатываем запрос…</div>}
 {result&&!busy&&<div className="map-status">{noCoverage?'Нет наблюдений за выбранный период и область':!result.active_fires.features.length&&!result.burned_areas.features.length?'Наблюдения есть, обнаружений нет':`${result.summary.active_detection_count} термоточек · ${result.summary.burned_contour_count} контуров`}{!noCoverage&&(result.coverage.af.status!=='full'||result.coverage.bs.status!=='full')&&<span> · Неполное покрытие AF/BS</span>}</div>}
 <div className="map-legend"><span className="eyebrow">СТЕПЕНЬ ПОРАЖЕНИЯ</span><div>{[1,2,3].map(k=><span key={k}><i style={{background:colors[k]}}/>{['','Слабая','Средняя','Сильная'][k]}</span>)}</div></div>
 </section></main>{upload&&<UploadDialog close={()=>setUpload(false)} done={id=>{setUpload(false);void load(id);}}/>}</div>;
}
