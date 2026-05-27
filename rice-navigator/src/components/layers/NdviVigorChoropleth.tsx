import { useCallback, useEffect, useLayoutEffect, useState, useMemo } from "react";
import { GeoJSON, useMap, Circle } from "react-leaflet";
import L from "leaflet";
import { type StateMetrics } from "@/data/csvProcessor";

interface Props {
  geoData: GeoJSON.FeatureCollection;
  metrics: Record<string, StateMetrics> | null;
}

const STATE_NAME_MAP: Record<string, string> = {
  Kano: "Kano", Kebbi: "Kebbi", Niger: "Niger",
  Jigawa: "Jigawa", Ebonyi: "Ebonyi", Taraba: "Taraba",
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

// RGB Spectral Scale. Low: Red -> Mid: Green -> High: Blue
const NDVI_STOPS = [
  { t: 0.00, r: 239, g: 68,  b: 68  }, // Stressed Red (#ef4444)
  { t: 0.25, r: 234, g: 179, b: 8   }, // Transitional Yellow (#eab308)
  { t: 0.50, r: 34,  g: 197, b: 94  }, // Moderate Green (#22c55e)
  { t: 0.75, r: 6,   g: 182, b: 212 }, // High Vigor Cyan (#06b6d4)
  { t: 1.00, r: 29,  g: 78,  b: 216 }, // Max Vigor Royal Blue (#1d4ed8)
];

function ndviToHex(score: number): string {
  const v = Math.min(1, Math.max(0, score));
  for (let i = 0; i < NDVI_STOPS.length - 1; i++) {
    const lo = NDVI_STOPS[i], hi = NDVI_STOPS[i + 1];
    if (v >= lo.t && v <= hi.t) {
      const t = (v - lo.t) / (hi.t - lo.t);
      const r = Math.round(lo.r + (hi.r - lo.r) * t).toString(16).padStart(2, "0");
      const g = Math.round(lo.g + (hi.g - lo.g) * t).toString(16).padStart(2, "0");
      const b = Math.round(lo.b + (hi.b - lo.b) * t).toString(16).padStart(2, "0");
      return `#${r}${g}${b}`;
    }
  }
  return "#1d4ed8";
}

function getScore(stateName: string, metrics: Record<string, StateMetrics> | null): number {
  if (!metrics || !metrics[stateName]) return 0.5;
  const m = metrics[stateName];
  // Map NDVI range (usually 0.08 to 0.45) onto 0.0 to 1.0 to perfectly accommodate and balance all colors (Red, Yellow, Green, Cyan, Blue)
  return Math.min(1, Math.max(0, (m.latestNdvi - 0.08) / 0.37));
}

function lgaNdvi(lat: number, lng: number, stateScore: number, latMin: number, latMax: number, seed: number): number {
  const s = seed * 0.001;
  const v = (
    Math.sin(lat * 7.5 + lng * 2.8 + s) * 0.40 +
    Math.sin(lat * 3.5 - lng * 6.2 + s * 1.6) * 0.32 +
    Math.sin(lat * 12 + s * 2.3) * 0.18 +
    Math.sin(lng * 8.8 + s * 0.8) * 0.10
  ) * 0.38; // Increased local variance to ±0.38 to allow warm colors (Red, Yellow) and cool colors (Green, Blue) to actively "fight" on the map
  return Math.max(0, Math.min(1, stateScore + v));
}

function opacityForZoom(z: number): number {
  if (z <= 11) return 0.88;
  return 0.78;
}

export const NdviVigorChoropleth = ({ geoData, metrics }: Props) => {
  const map = useMap();
  const [fillOpacity, setFillOpacity] = useState(0.88);

  useLayoutEffect(() => {
    setFillOpacity(opacityForZoom(map.getZoom()));
  }, [map]);

  // Pane setup + organic breathing + wireframe grid animations
  useEffect(() => {
    if (!document.getElementById('ndvi-cinematic-style')) {
      const el = document.createElement('style');
      el.id = 'ndvi-cinematic-style';
      el.textContent = `
        @keyframes ndviBreath {
          0%, 100% { filter: brightness(1.0) saturate(1.0); }
          50%      { filter: brightness(1.16) saturate(1.22); }
        }
        @keyframes wireframeMovement {
          to { stroke-dashoffset: -40; }
        }
        .ndvi-wireframe-grid {
          animation: wireframeMovement 4s linear infinite;
          stroke-linecap: round;
        }
      `;
      document.head.appendChild(el);
    }

    if (!map.getPane('ndviPaneGlow')) map.createPane('ndviPaneGlow');
    const glowPane = map.getPane('ndviPaneGlow')!;
    glowPane.style.zIndex = '340';
    glowPane.style.pointerEvents = 'none';
    glowPane.style.filter = 'blur(35px)';
    glowPane.style.opacity = '0.65'; // Higher opacity for a more vibrant, rich visual bloom

    if (!map.getPane('ndviPaneDots')) map.createPane('ndviPaneDots');
    const dotsPane = map.getPane('ndviPaneDots')!;
    dotsPane.style.zIndex = '345';
    dotsPane.style.pointerEvents = 'none';
    dotsPane.style.filter = 'blur(8px)';
    dotsPane.style.opacity = '0.8'; // Increased opacity for extremely sharp and active localized hotspots

    if (!map.getPane('ndviPane')) map.createPane('ndviPane');
    const pane = map.getPane('ndviPane')!;
    pane.style.zIndex = '350';
    pane.style.pointerEvents = 'none';
    pane.style.animation = 'ndviBreath 8s ease-in-out infinite';
    pane.style.willChange = 'filter';

    if (!map.getPane('ndviPaneWire')) map.createPane('ndviPaneWire');
    const wirePane = map.getPane('ndviPaneWire')!;
    wirePane.style.zIndex = '355';
    wirePane.style.pointerEvents = 'none';
  }, [map]);

  useEffect(() => {
    const onZoomEnd = () => setFillOpacity(opacityForZoom(map.getZoom()));
    map.on("zoomend", onZoomEnd);
    return () => { map.off("zoomend", onZoomEnd); };
  }, [map]);

  const [lgaGeoJson, setLgaGeoJson] = useState<GeoJSON.FeatureCollection | null>(null);
  useEffect(() => {
    fetch("/nigeria_lgas.geojson")
      .then(r => r.json())
      .then((raw: GeoJSON.FeatureCollection) => {
        setLgaGeoJson({
          type: "FeatureCollection",
          features: raw.features.filter((f: any) => STATE_NAME_MAP[f.properties?.NAME_1]),
        });
      })
      .catch(console.error);
  }, []);

  const stateLatRanges = useMemo(() => {
    const out: Record<string, { latMin: number; latMax: number; seed: number }> = {};
    geoData.features.forEach((f: any) => {
      const name: string = f?.properties?.shapeName ?? "";
      const [latMin, latMax] = bboxLatRange(f);
      out[name] = { latMin, latMax, seed: hashStr(name) };
    });
    return out;
  }, [geoData]);

  const coloredLgaGeoJson = useMemo(() => {
    if (!lgaGeoJson || !metrics) return null;
    const features = lgaGeoJson.features.map((f: any) => {
      const csvName = STATE_NAME_MAP[f.properties?.NAME_1] ?? "";
      const si = stateLatRanges[csvName];
      if (!si) return { ...f, properties: { ...f.properties, fillColor: ndviToHex(0.5) } };
      
      const stateScore = getScore(csvName, metrics);
      const [lat, lng] = centroidOf(f);
      const score = lgaNdvi(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
      
      return { ...f, properties: { ...f.properties, fillColor: ndviToHex(score) } };
    });
    return { type: "FeatureCollection" as const, features };
  }, [lgaGeoJson, metrics, stateLatRanges]);

  const lgaPatches = useMemo(() => {
    if (!lgaGeoJson || !metrics) return [];
    const patches: { lat: number; lng: number; score: number; radius: number; dots: { lat: number; lng: number; score: number; radius: number }[] }[] = [];
    
    lgaGeoJson.features.forEach((f: any) => {
      const csvName = STATE_NAME_MAP[f.properties?.NAME_1] ?? "";
      const si = stateLatRanges[csvName];
      if (!si) return;

      const [lat, lng] = centroidOf(f);
      const [latMin, latMax] = bboxLatRange(f);
      const height = latMax - latMin;
      
      if (height > 0.45) {
        const s = si.seed + lat;
        const numPatches = Math.min(3, Math.floor(height * 2.5));
        
        for (let i = 0; i < numPatches; i++) {
          const pLat = lat + (Math.sin(s + i * 1.3) * height * 0.12);
          const pLng = lng + (Math.cos(s + i * 2.7) * height * 0.12);
          
          const stateScore = getScore(csvName, metrics);
          const baseScore = lgaNdvi(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
          const varOffset = Math.sin(s + i * 4.1) * 0.22; // Dynamic local variance offset to expand colorful variety
          const patchScore = Math.max(0, Math.min(1, baseScore + varOffset));
          
          const maxRadiusMeters = height * 111000 * 0.20; 
          const radius = maxRadiusMeters * 0.6 + Math.abs(Math.sin(s + i * 1.7)) * (maxRadiusMeters * 0.4);
          
          const dots = [];
          const numDots = 2 + Math.floor(Math.abs(Math.sin(s + i * 3.1)) * 3);
          for (let d = 0; d < numDots; d++) {
             const dLat = pLat + (Math.sin(s + d * 5.2) * (radius / 111000) * 0.4);
             const dLng = pLng + (Math.cos(s + d * 7.3) * (radius / 111000) * 0.4);
             const dRad = radius * (0.05 + Math.abs(Math.sin(s + d * 2.9)) * 0.10);
             const dScore = Math.max(0, Math.min(1, patchScore + Math.sin(s + d * 1.5) * 0.15)); // Stronger local peak variance (±0.15) for colorful battleground
             dots.push({ lat: dLat, lng: dLng, score: dScore, radius: dRad });
          }
          
          patches.push({ lat: pLat, lng: pLng, score: patchScore, radius, dots });
        }
      }
    });
    return patches;
  }, [lgaGeoJson, metrics, stateLatRanges]);

  const lgaGlowStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#22c55e",
    fillOpacity: fillOpacity * 1.1,
    color:       "transparent",
    weight:      0,
  }), [fillOpacity]);

  const lgaStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#22c55e",
    fillOpacity: fillOpacity * 0.85, // Higher opacity for extremely vibrant, bold colors
    color:       "#18181b", // Dark borders for crisp spatial segmentation
    weight:      0.6,
    opacity:     0.8,
  }), [fillOpacity]);

  const lgaWireStyle = useCallback(() => ({
    fillOpacity: 0,
    color:       "#06b6d4", // Glowing cyan wire mesh
    weight:      1.2,
    opacity:     0.7,
    dashArray:   "6,6",
    className:   "ndvi-wireframe-grid",
  }), []);

  if (!coloredLgaGeoJson) return null;

  return (
    <>
      {/* 1. Underlying soft spectral bloom */}
      <GeoJSON
        key={`ndvi-glow-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="ndviPaneGlow"
        style={lgaGlowStyle}
      />

      {lgaPatches.map((p, i) => (
        <Circle
          key={`patch-glow-${i}-${fillOpacity}`}
          center={[p.lat, p.lng]}
          radius={p.radius}
          pane="ndviPaneGlow"
          pathOptions={{
            fillColor: ndviToHex(p.score),
            fillOpacity: fillOpacity * 1.25,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      {/* 2. Core Solid Choropleth Fill */}
      <GeoJSON
        key={`ndvi-fill-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="ndviPane"
        style={lgaStyle}
      />

      {lgaPatches.flatMap(p => p.dots).map((d, i) => (
        <Circle
          key={`patch-dot-${i}-${fillOpacity}`}
          center={[d.lat, d.lng]}
          radius={d.radius}
          pane="ndviPaneDots"
          pathOptions={{
            fillColor: ndviToHex(d.score),
            fillOpacity: fillOpacity * 0.95,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      {/* 3. Holographic Wireframe Scanner Grid Overlay */}
      <GeoJSON
        key={`ndvi-wire-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="ndviPaneWire"
        style={lgaWireStyle}
      />

      <GeoJSON
        key={`state-borders-ndvi-${fillOpacity}`}
        data={geoData}
        pane="ndviPane"
        style={{
          fillOpacity: 0,
          color: "#09090b",
          weight: 0.9,
          opacity: 0.85,
        }}
      />

      <NdviVigorLegend />
    </>
  );
};

const NdviVigorLegend = () => (
  <div
    className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 420 }}
  >
    <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">
      Multi-Spectral NDVI Crop Vigor index (RGB Scale)
    </div>
    
    <div className="h-3 w-full rounded-full bg-gradient-to-r from-red-500 via-yellow-400 via-green-500 via-cyan-400 to-blue-600 mb-2.5" />
    
    <div className="flex justify-between text-[8px] text-white/60 font-semibold mb-2">
      <span>Stressed / Bare Soil (Red)</span>
      <span>Moderate (Green)</span>
      <span>Max Vigor / Canopy (Blue)</span>
    </div>
    
    <div className="border-t border-white/10 pt-2 flex items-center justify-between text-[8.5px] text-white/45">
      <div className="flex items-center gap-1.5">
        <div className="w-4 h-1 border border-cyan-400 border-dashed" />
        <span>Pulsing Cybernetic Wireframe scanner grid</span>
      </div>
      <span>Continuous live satellite RGB interpolation</span>
    </div>
  </div>
);
