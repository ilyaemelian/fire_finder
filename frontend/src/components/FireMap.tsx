import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON, Rectangle, useMap, useMapEvents, CircleMarker, Popup, AttributionControl } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import type { Polygon } from 'geojson';
import type { Dataset, Result } from '../types';
import { polygon } from '../types';
export const colors:Record<number,string>={1:'#e6b84a',2:'#e67e36',3:'#bf493b'};
function View({dataset}:{dataset:Dataset}){const map=useMap();useEffect(()=>{const [w,s,e,n]=dataset.bounds;map.fitBounds([[s,w],[n,e]],{padding:[36,36]});},[dataset.id,map]);return null;}
function Draw({enabled,onSelect,onStep}:{enabled:boolean;onSelect:(p:Polygon)=>void;onStep:(v:boolean)=>void}){
 const map=useMap();const [first,setFirst]=useState<LatLngTuple|null>(null);const [last,setLast]=useState<LatLngTuple|null>(null);
 useEffect(()=>{setFirst(null);setLast(null);onStep(false);map.getContainer().style.cursor=enabled?'crosshair':'';return ()=>{map.getContainer().style.cursor='';};},[enabled,map]);
 useMapEvents({click(e){if(!enabled)return;const current:LatLngTuple=[e.latlng.lat,e.latlng.lng];if(!first){setFirst(current);setLast(current);onStep(true);}else{const s=Math.min(first[0],current[0]),n=Math.max(first[0],current[0]),w=Math.min(first[1],current[1]),east=Math.max(first[1],current[1]);if(n-s<.00001||east-w<.00001)return;onSelect(polygon([w,s,east,n]));setFirst(null);setLast(null);}},mousemove(e){if(enabled&&first)setLast([e.latlng.lat,e.latlng.lng]);}});
 return first&&last?<Rectangle bounds={[first,last]} pathOptions={{color:'#245bba',weight:2,dashArray:'6 5',fillOpacity:.06}}/>:null;
}
export default function FireMap({dataset,result,selection,drawing,onSelect,showAF,showBS}:{dataset:Dataset;result:Result|null;selection:Polygon|null;drawing:boolean;onSelect:(p:Polygon)=>void;showAF:boolean;showBS:boolean}){
 const [second,setSecond]=useState(false);
 return <><MapContainer center={[46,39]} zoom={10} className="map" zoomControl={true} attributionControl={false}>
 <AttributionControl position="bottomright" prefix={'<a href="https://leafletjs.com" title="Библиотека интерактивных карт">Leaflet</a>'}/>
 <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"/>
 <View dataset={dataset}/><Draw enabled={drawing} onSelect={onSelect} onStep={setSecond}/>
 {showBS&&result&&<GeoJSON key={result.result_id+'bs'} data={result.burned_areas} style={f=>({color:colors[f?.properties?.severity]||'#666',weight:1.5,fillOpacity:.57})} onEachFeature={(f,l)=>{const p=f.properties;const box=document.createElement('div');const title=document.createElement('strong');title.textContent='Контур гари';box.append(title);for(const value of [`${p.contour_id}`,`Степень: ${['','слабая','средняя','сильная'][p.severity]}`,`Площадь: ${Number(p.area_ha).toLocaleString('ru-RU',{maximumFractionDigits:2})} га`,`Снимки: ${p.date_pre} → ${p.date_post}`,`Модель: ${p.model_version}`,dataset.data_mode==='synthetic'?'Синтетический пример':dataset.source]){const line=document.createElement('div');line.textContent=value;box.append(line);}l.bindPopup(box);}}/>}
 {showAF&&result&&result.active_fires.features.map(f=>{if(f.geometry.type!=='Point')return null;const [lng,lat]=f.geometry.coordinates;return <CircleMarker key={f.properties?.detection_id} center={[lat,lng]} radius={5} pathOptions={{color:'#fff',weight:1.5,fillColor:'#e75537',fillOpacity:1}}><Popup><strong>Активное горение</strong><br/>{f.properties?.acquisition_datetime}<br/>Модель: {f.properties?.model_version}<br/>{dataset.data_mode==='synthetic'?'Синтетическое обнаружение':dataset.source}</Popup></CircleMarker>;})}
 {selection&&<GeoJSON key={JSON.stringify(selection)} data={selection} style={{color:'#245bba',weight:2,dashArray:'6 4',fillOpacity:0}}/>}
 </MapContainer>{drawing&&<div className="draw-hint">{second?'Нажмите на противоположный угол':'Нажмите на первый угол области'} · Esc — отмена</div>}</>;
}
