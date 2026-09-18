import type { FeatureCollection, Polygon } from 'geojson';
export type Dataset = {id:string;title:string;source:string;data_mode:'synthetic'|'real';bounds:[number,number,number,number];date_from:string;date_to:string;observations:FeatureCollection};
export type Query = {dataset_id:string;geometry:Polygon;date_from:string;date_to:string};
export type Result = {result_id:string;query:Query;data_mode:string;source:string;coverage:Record<'af'|'bs',{has_observations:boolean;fraction:number;status:string}>;summary:{total_burned_area_ha:number;area_by_severity_ha:Record<string,number>;active_detection_count:number;burned_contour_count:number};active_fires:FeatureCollection;burned_areas:FeatureCollection};
export function polygon(b:number[]):Polygon {const [w,s,e,n]=b;return {type:'Polygon',coordinates:[[[w,s],[e,s],[e,n],[w,n],[w,s]]]};}
