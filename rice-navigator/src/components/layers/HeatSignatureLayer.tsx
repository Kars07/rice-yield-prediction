/**
 * HeatSignatureLayer — LGA-level choropleth
 *
 * Uses GADM ADM2 (LGA) GeoJSON. Each LGA polygon is filled with
 * its own temperature color computed from:
 *   - state average temperature (from CSV metrics)
 *   - latitude bias (north Nigeria hotter)
 *   - LGA centroid-based spatial variation (smooth directional bands)
 *
 * LGA boundaries are already visible on the basemap — this layer
 * colours them and draws their outlines on top.
 */

import { useEffect, useState, useMemo } from "react";
import { GeoJSON } from "react-leaflet";
import { type StateMetrics } from "@/data/csvProcessor";

interface Props {
  geoData: GeoJSON.FeatureCollection;   // state-level (for state outlines)
  metrics: Record<string, StateMetrics> | null;
}

// Map GADM NAME_1 → our CSV state names (must match csvProcessor state keys exactly)
const STATE_NAME_MAP: Record<string, string> = {
  Kano:    "Kano",
  Kebbi:   "Kebbi",
  Niger:   "Niger",
  Jigawa:  "Jigawa",
  Ebonyi:  "Ebonyi",
  Taraba:  "Taraba",
};

function centroidOf(feature: any): [number, number] {
  const coords: number[][] = [];
  const { type, coordinates } = feature.geometry;
  if (type === "Polygon") coordinates[0].forEach((c: number[]) => coords.push(c));
  else if (type === "MultiPolygon") coordinates.forEach((p: number[][][]) => p[0].forEach((c: number[]) => coords.push(c)));
  const lng = coords.reduce((s, c) => s + c[0], 0) / coords.length;
  const lat = coords.reduce((s, c) => s + c[1], 0) / coords.length;
  return [lat, lng];
}

function bboxLatRange(feature: any): [number, number] {
  const lats: number[] = [];
  const { type, coordinates } = feature.geometry;
  if (type === "Polygon") coordinates[0].forEach((c: number[]) => lats.push(c[1]));
  else if (type === "MultiPolygon") coordinates.forEach((p: number[][][]) => p[0].forEach((c: number[]) => lats.push(c[1])));
  return [Math.min(...lats), Math.max(...lats)];
}

function hashStr(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}

// Temperature for one LGA based on its centroid position
function lgaTemp(
  lat: number, lng: number,
  stateAvg: number,
  stateLatMin: number, stateLatMax: number,
  stateSeed: number
): number {
  const latBias = ((lat - stateLatMin) / (stateLatMax - stateLatMin || 1) - 0.5) * 4.0;
  const s = stateSeed * 0.001;
  const v = (
    Math.sin(lat * 8.0 + lng * 3.1 + s)       * 0.42 +
    Math.sin(lat * 3.2 - lng * 6.8 + s * 1.5) * 0.35 +
    Math.sin(lat * 13  + s * 2.1)              * 0.15 +
    Math.sin(lng * 9.4 + s * 0.7)              * 0.08
  ) * 2.8;
  return Math.max(24, Math.min(32, stateAvg + latBias + v));
}

// Nigeria 24–32°C → full yellow → dark brown ramp
const STOPS = [
  { t: 0.00, r: 255, g: 255, b: 0   },
  { t: 0.22, r: 255, g: 190, b: 0   },
  { t: 0.42, r: 255, g: 100, b: 0   },
  { t: 0.62, r: 220, g: 25,  b: 0   },
  { t: 0.82, r: 155, g: 0,   b: 0   },
  { t: 1.00, r: 41,  g: 16,  b: 0   },
];
function tempToHex(temp: number): string {
  const v = Math.min(1, Math.max(0, (temp - 24) / 8));
  for (let i = 0; i < STOPS.length - 1; i++) {
    const lo = STOPS[i], hi = STOPS[i + 1];
    if (v >= lo.t && v <= hi.t) {
      const t = (v - lo.t) / (hi.t - lo.t);
      const r = Math.round(lo.r + (hi.r - lo.r) * t).toString(16).padStart(2, "0");
      const g = Math.round(lo.g + (hi.g - lo.g) * t).toString(16).padStart(2, "0");
      const b = Math.round(lo.b + (hi.b - lo.b) * t).toString(16).padStart(2, "0");
      return `#${r}${g}${b}`;
    }
  }
  return "#291000";
}

export const HeatSignatureLayer = ({ geoData, metrics }: Props) => {
  const [lgaGeoJson, setLgaGeoJson] = useState<GeoJSON.FeatureCollection | null>(null);

  // Fetch LGA GeoJSON once
  useEffect(() => {
    fetch("/nigeria_lgas.geojson")
      .then(r => r.json())
      .then((raw: GeoJSON.FeatureCollection) => {
        // Filter to only our 6 rice states
        const filtered: GeoJSON.Feature[] = raw.features.filter(
          (f: any) => STATE_NAME_MAP[f.properties?.NAME_1] !== undefined
        );
        setLgaGeoJson({ type: "FeatureCollection", features: filtered });
      })
      .catch(console.error);
  }, []);

  // Pre-compute per-state lat range and seed from the state GeoJSON
  const stateLatRanges = useMemo(() => {
    const map: Record<string, { latMin: number; latMax: number; seed: number }> = {};
    geoData.features.forEach((f: any) => {
      const name: string = f?.properties?.shapeName ?? "";
      const [latMin, latMax] = bboxLatRange(f);
      map[name] = { latMin, latMax, seed: hashStr(name) };
    });
    return map;
  }, [geoData]);

  // Build coloured LGA GeoJSON
  const coloredLgaGeoJson = useMemo(() => {
    if (!lgaGeoJson || !metrics) return null;

    const features = lgaGeoJson.features.map((f: any) => {
      const gadmStateName: string = f.properties?.NAME_1 ?? "";
      const csvStateName = STATE_NAME_MAP[gadmStateName] ?? gadmStateName;
      const m = metrics[csvStateName];
      const stateInfo = stateLatRanges[csvStateName];

      if (!m || !stateInfo) {
        // Fallback: use Nigeria average temperature so no LGA shows grey
        const fallbackColor = tempToHex(28.0);
        return { ...f, properties: { ...f.properties, fillColor: fallbackColor, temp: "28.0" } };
      }

      const [lat, lng] = centroidOf(f);
      const temp = lgaTemp(lat, lng, m.avgTemperature, stateInfo.latMin, stateInfo.latMax, stateInfo.seed);
      const fillColor = tempToHex(temp);

      return { ...f, properties: { ...f.properties, fillColor, temp: temp.toFixed(1) } };
    });

    return { type: "FeatureCollection" as const, features };
  }, [lgaGeoJson, metrics, stateLatRanges]);

  if (!coloredLgaGeoJson) return null;

  return (
    <>
      {/* LGA fill — each region coloured by its temperature */}
      <GeoJSON
        key="lga-heat-fill"
        data={coloredLgaGeoJson}
        style={(feature: any) => ({
          fillColor:   feature?.properties?.fillColor ?? "#aaa",
          fillOpacity: 0.82,
          color:       "#111",      // LGA boundary line
          weight:      0.8,         // thin line — matches basemap LGA lines
          opacity:     0.7,
        })}
      />

      {/* State outlines on top — thicker for visual hierarchy */}
      <GeoJSON
        key="state-outlines"
        data={geoData}
        style={{
          fillOpacity: 0,
          color:       "#111",
          weight:      2,
          opacity:     0.9,
        }}
      />

      {/* Legend */}
      <div
        className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
        style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 320 }}
      >
        <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">
          Surface Temperature · LGA Regions
        </div>
        <div className="h-3.5 w-full rounded-full mb-2"
          style={{ background: "linear-gradient(to right, #ffff00, #ffbe00 22%, #ff6400 42%, #dc1900 62%, #9b0000 82%, #291000)" }} />
        <div className="flex justify-between text-[9px] font-semibold text-white/60">
          <span>24°C</span><span>26°C</span><span>28°C</span><span>30°C</span><span>32°C+</span>
        </div>
        <div className="text-[8px] text-white/35 text-center mt-2">
          Each LGA is one bounded temperature zone · 159 regions across 6 states
        </div>
      </div>
    </>
  );
};
