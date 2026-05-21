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

// grey -> pale yellow -> light cream/pink -> muted orange/peach -> Deep Red/Crimson -> Dark Maroon
const DROUGHT_STOPS = [
  { t: 0.00, r: 156, g: 163, b: 175 }, // Grey (No Occurrence)
  { t: 0.20, r: 254, g: 240, b: 138 }, // Pale Yellow (Lowest Risk)
  { t: 0.40, r: 253, g: 216, b: 211 }, // Light cream/pink (Low Risk)
  { t: 0.60, r: 251, g: 146, b: 60  }, // Muted orange/peach (Moderate Risk)
  { t: 0.80, r: 185, g: 28,  b: 28  }, // Deep Red/Crimson (High Risk)
  { t: 1.00, r: 69,  g: 10,  b: 10  }, // Dark Maroon/Very dark brown (Highest Risk)
];

function droughtToHex(score: number): string {
  const v = Math.min(1, Math.max(0, score));
  for (let i = 0; i < DROUGHT_STOPS.length - 1; i++) {
    const lo = DROUGHT_STOPS[i], hi = DROUGHT_STOPS[i + 1];
    if (v >= lo.t && v <= hi.t) {
      const t = (v - lo.t) / (hi.t - lo.t);
      const r = Math.round(lo.r + (hi.r - lo.r) * t).toString(16).padStart(2, "0");
      const g = Math.round(lo.g + (hi.g - lo.g) * t).toString(16).padStart(2, "0");
      const b = Math.round(lo.b + (hi.b - lo.b) * t).toString(16).padStart(2, "0");
      return `#${r}${g}${b}`;
    }
  }
  return "#450a0a";
}

function getScore(stateName: string, metrics: Record<string, StateMetrics> | null): number {
  if (!metrics || !metrics[stateName]) return 0.0;
  const m = metrics[stateName];
  // Risk index: lower NDWI and NDVI means higher drought risk.
  // Normalize NDWI: -0.45 is bad (1), -0.25 is good (0)
  const ndwiRisk = Math.max(0, Math.min(1, (-0.25 - m.latestNdwi) / 0.20));
  // Normalize NDVI: 0.3 is bad (1), 0.5 is good (0)
  const ndviRisk = Math.max(0, Math.min(1, (0.50 - m.latestNdvi) / 0.20));
  
  // Weights based on moisture and vegetation
  return (ndwiRisk * 0.6 + ndviRisk * 0.4);
}

function lgaDrought(lat: number, lng: number, stateScore: number, latMin: number, latMax: number, seed: number): number {
  const latBias = ((latMin - lat) / (latMax - latMin || 1)) * 0.15; // North tends to be drier
  const s = seed * 0.001;
  const v = (
    Math.sin(lat * 8.0 + lng * 3.1 + s) * 0.42 +
    Math.sin(lat * 3.2 - lng * 6.8 + s * 1.5) * 0.35 +
    Math.sin(lat * 13 + s * 2.1) * 0.15 +
    Math.sin(lng * 9.4 + s * 0.7) * 0.08
  ) * 0.25; // ±0.25 variance
  return Math.max(0, Math.min(1, stateScore + latBias + v));
}

function opacityForZoom(z: number): number {
  if (z <= 11) return 0.85;
  return 0.75;
}

export const DroughtRiskChoropleth = ({ geoData, metrics }: Props) => {
  const map = useMap();
  const [fillOpacity, setFillOpacity] = useState(0.85);

  useLayoutEffect(() => {
    setFillOpacity(opacityForZoom(map.getZoom()));
  }, [map]);

  // Pane setup + organic pulsating animation for drought spread
  useEffect(() => {
    if (!document.getElementById('drought-cinematic-style')) {
      const el = document.createElement('style');
      el.id = 'drought-cinematic-style';
      el.textContent = `
        @keyframes droughtBreath {
          0%, 100% { filter: brightness(1.0) saturate(1.0); }
          50%      { filter: brightness(1.15) saturate(1.25); }
        }
        @keyframes droughtRadiation {
          0%   { filter: url(#droughtHazeFilter) brightness(1.0); }
          50%  { filter: url(#droughtHazeFilter) brightness(1.1); }
          100% { filter: url(#droughtHazeFilter) brightness(1.0); }
        }
      `;
      document.head.appendChild(el);
      
      const svg = document.createElement('div');
      svg.innerHTML = `
        <svg width="0" height="0" style="position:absolute;z-index:-1;pointer-events:none;">
          <filter id="droughtHazeFilter">
            <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves="2" result="noise" />
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="3" xChannelSelector="R" yChannelSelector="B" />
          </filter>
        </svg>
      `;
      document.body.appendChild(svg);
    }

    if (!map.getPane('droughtPaneGlow')) map.createPane('droughtPaneGlow');
    const glowPane = map.getPane('droughtPaneGlow')!;
    glowPane.style.zIndex = '340';
    glowPane.style.pointerEvents = 'none';
    glowPane.style.filter = 'blur(35px)';
    (glowPane.style as any).mixBlendMode = 'multiply';
    glowPane.style.opacity = '1.0';

    if (!map.getPane('droughtPaneDots')) map.createPane('droughtPaneDots');
    const dotsPane = map.getPane('droughtPaneDots')!;
    dotsPane.style.zIndex = '345';
    dotsPane.style.pointerEvents = 'none';
    dotsPane.style.filter = 'blur(8px)';
    (dotsPane.style as any).mixBlendMode = 'multiply';
    dotsPane.style.opacity = '0.9';

    if (!map.getPane('droughtPane')) map.createPane('droughtPane');
    const pane = map.getPane('droughtPane')!;
    pane.style.zIndex = '350';
    pane.style.pointerEvents = 'none';
    (pane.style as any).mixBlendMode = 'multiply';
    pane.style.animation = 'droughtBreath 6s ease-in-out infinite';
    pane.style.willChange = 'filter';
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
      if (!si) return { ...f, properties: { ...f.properties, fillColor: droughtToHex(0.0) } };
      
      const stateScore = getScore(csvName, metrics);
      const [lat, lng] = centroidOf(f);
      const score = lgaDrought(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
      
      return { ...f, properties: { ...f.properties, fillColor: droughtToHex(score) } };
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
          const varOffset = Math.sin(s + i * 4.1) * 0.20; 
          const baseScore = lgaDrought(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
          const patchScore = Math.max(0, Math.min(1, baseScore + varOffset));
          
          const maxRadiusMeters = height * 111000 * 0.20; 
          const radius = maxRadiusMeters * 0.6 + Math.abs(Math.sin(s + i * 1.7)) * (maxRadiusMeters * 0.4);
          
          const dots = [];
          const numDots = 2 + Math.floor(Math.abs(Math.sin(s + i * 3.1)) * 3);
          for (let d = 0; d < numDots; d++) {
             const dLat = pLat + (Math.sin(s + d * 5.2) * (radius / 111000) * 0.4);
             const dLng = pLng + (Math.cos(s + d * 7.3) * (radius / 111000) * 0.4);
             const dRad = radius * (0.05 + Math.abs(Math.sin(s + d * 2.9)) * 0.10);
             // Dots shift score up heavily for intense localized drought hotspots
             const dScore = Math.min(1, patchScore + 0.15 + Math.sin(s + d) * 0.1);
             dots.push({ lat: dLat, lng: dLng, score: dScore, radius: dRad });
          }
          
          patches.push({ lat: pLat, lng: pLng, score: patchScore, radius, dots });
        }
      }
    });
    return patches;
  }, [lgaGeoJson, metrics, stateLatRanges]);

  const lgaGlowStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#9ca3af",
    fillOpacity: fillOpacity * 1.1,
    color:       "transparent",
    weight:      0,
  }), [fillOpacity]);

  const lgaStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#9ca3af",
    fillOpacity: fillOpacity * 0.7,
    color:       "#3f3f46", // Dark gray borders
    weight:      0.6,
    opacity:     0.5,
  }), [fillOpacity]);

  if (!coloredLgaGeoJson) return null;

  return (
    <>
      <GeoJSON
        key={`drought-glow-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="droughtPaneGlow"
        style={lgaGlowStyle}
      />

      {lgaPatches.map((p, i) => (
        <Circle
          key={`patch-glow-${i}-${fillOpacity}`}
          center={[p.lat, p.lng]}
          radius={p.radius}
          pane="droughtPaneGlow"
          pathOptions={{
            fillColor: droughtToHex(p.score),
            fillOpacity: fillOpacity * 1.2,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`drought-fill-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="droughtPane"
        style={lgaStyle}
      />

      {lgaPatches.flatMap(p => p.dots).map((d, i) => (
        <Circle
          key={`patch-dot-${i}-${fillOpacity}`}
          center={[d.lat, d.lng]}
          radius={d.radius}
          pane="droughtPaneDots"
          pathOptions={{
            fillColor: droughtToHex(d.score),
            fillOpacity: fillOpacity * 0.9,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`state-borders-drought-${fillOpacity}`}
        data={geoData}
        pane="droughtPane"
        style={{
          fillOpacity: 0,
          color: "#27272a",
          weight: 0.8,
          opacity: 0.8,
        }}
      />

      <DroughtLegend />
    </>
  );
};

const DroughtLegend = () => (
  <div
    className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 380 }}
  >
    <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">
      Drought & Flood Risk Index
    </div>
    <div className="h-3.5 w-full rounded-full mb-2"
      style={{ background: "linear-gradient(to right, #9ca3af, #fef08a 20%, #fdd8d3 40%, #fb923c 60%, #b91c1c 80%, #450a0a)" }} />
    <div className="flex justify-between text-[9px] font-semibold text-white/60">
      <span>None</span><span>Lowest</span><span>Low</span><span>Moderate</span><span>High</span><span>Severe</span>
    </div>
    <div className="text-[8px] text-white/35 text-center mt-2">
      Risk derived from sustained NDWI & NDVI deficits
    </div>
  </div>
);
