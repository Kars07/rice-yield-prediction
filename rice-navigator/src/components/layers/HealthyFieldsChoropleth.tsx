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

// 0 to 1 scale. 0 = Barren/Dry (yellow-brown), 1 = Lush Dense Green
const HEALTH_STOPS = [
  { t: 0.00, r: 245, g: 222, b: 179 }, // Barren/Dry (wheat/beige)
  { t: 0.20, r: 212, g: 225, b: 87 },  // Stressed/Sparse (yellow-green)
  { t: 0.40, r: 156, g: 204, b: 101 }, // Light Green
  { t: 0.60, r: 102, g: 187, b: 106 }, // Normal Green (healthy)
  { t: 0.80, r: 67,  g: 160, b: 71 },  // Very Healthy (vivid green)
  { t: 1.00, r: 27,  g: 94,  b: 32 },  // Dense Canopy (dark forest green)
];

function healthToHex(score: number): string {
  const v = Math.min(1, Math.max(0, score));
  for (let i = 0; i < HEALTH_STOPS.length - 1; i++) {
    const lo = HEALTH_STOPS[i], hi = HEALTH_STOPS[i + 1];
    if (v >= lo.t && v <= hi.t) {
      const t = (v - lo.t) / (hi.t - lo.t);
      const r = Math.round(lo.r + (hi.r - lo.r) * t).toString(16).padStart(2, "0");
      const g = Math.round(lo.g + (hi.g - lo.g) * t).toString(16).padStart(2, "0");
      const b = Math.round(lo.b + (hi.b - lo.b) * t).toString(16).padStart(2, "0");
      return `#${r}${g}${b}`;
    }
  }
  return "#1b5e20";
}

function getScore(stateName: string, metrics: Record<string, StateMetrics> | null): number {
  if (!metrics || !metrics[stateName]) return 0.5;
  // Use NDVI directly for health score. Normalize 0.1 to 0.7 into 0 to 1
  const ndvi = metrics[stateName].latestNdvi;
  return Math.min(1, Math.max(0, (ndvi - 0.1) / 0.6));
}

function lgaHealth(lat: number, lng: number, stateScore: number, latMin: number, latMax: number, seed: number): number {
  const latBias = ((latMin - lat) / (latMax - latMin || 1)) * 0.1; // Slight regional bias
  const s = seed * 0.001;
  const v = (
    Math.sin(lat * 8.0 + lng * 3.1 + s) * 0.42 +
    Math.sin(lat * 3.2 - lng * 6.8 + s * 1.5) * 0.35 +
    Math.sin(lat * 13 + s * 2.1) * 0.15 +
    Math.sin(lng * 9.4 + s * 0.7) * 0.08
  ) * 0.3; // ±0.3 variance
  return Math.max(0, Math.min(1, stateScore + latBias + v));
}

function opacityForZoom(z: number): number {
  if (z <= 11) return 0.90;
  return 0.78;
}

export const HealthyFieldsChoropleth = ({ geoData, metrics }: Props) => {
  const map = useMap();
  const [fillOpacity, setFillOpacity] = useState(0.90);

  useLayoutEffect(() => {
    setFillOpacity(opacityForZoom(map.getZoom()));
  }, [map]);

  // Pane setup + organic growth/wind animation
  useEffect(() => {
    if (!document.getElementById('health-cinematic-style')) {
      const el = document.createElement('style');
      el.id = 'health-cinematic-style';
      el.textContent = `
        @keyframes healthBreath {
          0%, 100% { filter: brightness(1.0) saturate(1.0); }
          50%      { filter: brightness(1.06) saturate(1.15); }
        }
        @keyframes windFlowMovement {
          0%   { background-position: 0px 0px, 0px 0px; }
          100% { background-position: 600px 300px, 1200px 600px; }
        }
      `;
      document.head.appendChild(el);
    }

    if (!map.getPane('healthPaneGlow')) map.createPane('healthPaneGlow');
    const glowPane = map.getPane('healthPaneGlow')!;
    glowPane.style.zIndex = '340';
    glowPane.style.pointerEvents = 'none';
    glowPane.style.filter = 'blur(35px)';
    (glowPane.style as any).mixBlendMode = 'multiply';
    glowPane.style.opacity = '1.0';

    if (!map.getPane('healthPaneDots')) map.createPane('healthPaneDots');
    const dotsPane = map.getPane('healthPaneDots')!;
    dotsPane.style.zIndex = '345';
    dotsPane.style.pointerEvents = 'none';
    dotsPane.style.filter = 'blur(8px)';
    (dotsPane.style as any).mixBlendMode = 'multiply';
    dotsPane.style.opacity = '0.9';

    if (!map.getPane('healthPane')) map.createPane('healthPane');
    const pane = map.getPane('healthPane')!;
    pane.style.zIndex = '350';
    pane.style.pointerEvents = 'none';
    (pane.style as any).mixBlendMode = 'multiply';
    pane.style.animation = 'healthBreath 5s ease-in-out infinite';
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
      if (!si) return { ...f, properties: { ...f.properties, fillColor: healthToHex(0.5) } };
      
      const stateScore = getScore(csvName, metrics);
      const [lat, lng] = centroidOf(f);
      const score = lgaHealth(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
      
      return { ...f, properties: { ...f.properties, fillColor: healthToHex(score) } };
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
          const healthVariance = Math.sin(s + i * 4.1) * 0.25; 
          const baseScore = lgaHealth(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
          const patchScore = Math.max(0, Math.min(1, baseScore + healthVariance));
          
          const maxRadiusMeters = height * 111000 * 0.20; 
          const radius = maxRadiusMeters * 0.6 + Math.abs(Math.sin(s + i * 1.7)) * (maxRadiusMeters * 0.4);
          
          const dots = [];
          const numDots = 2 + Math.floor(Math.abs(Math.sin(s + i * 3.1)) * 3);
          for (let d = 0; d < numDots; d++) {
             const dLat = pLat + (Math.sin(s + d * 5.2) * (radius / 111000) * 0.4);
             const dLng = pLng + (Math.cos(s + d * 7.3) * (radius / 111000) * 0.4);
             const dRad = radius * (0.05 + Math.abs(Math.sin(s + d * 2.9)) * 0.10);
             // Shift score up for intense dense forest cluster
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
    fillColor:   feature?.properties?.fillColor ?? "#4caf50",
    fillOpacity: fillOpacity * 1.1,
    color:       "transparent",
    weight:      0,
  }), [fillOpacity]);

  const lgaStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#4caf50",
    fillOpacity: fillOpacity * 0.65,
    color:       "#052e16", // Dark green border
    weight:      0.5,
    opacity:     0.72,
  }), [fillOpacity]);

  if (!coloredLgaGeoJson) return null;

  return (
    <>
      <GeoJSON
        key={`health-glow-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="healthPaneGlow"
        style={lgaGlowStyle}
      />

      {lgaPatches.map((p, i) => (
        <Circle
          key={`patch-glow-${i}-${fillOpacity}`}
          center={[p.lat, p.lng]}
          radius={p.radius}
          pane="healthPaneGlow"
          pathOptions={{
            fillColor: healthToHex(p.score),
            fillOpacity: fillOpacity * 1.2,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`health-fill-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="healthPane"
        style={lgaStyle}
      />

      {lgaPatches.flatMap(p => p.dots).map((d, i) => (
        <Circle
          key={`patch-dot-${i}-${fillOpacity}`}
          center={[d.lat, d.lng]}
          radius={d.radius}
          pane="healthPaneDots"
          pathOptions={{
            fillColor: healthToHex(d.score),
            fillOpacity: fillOpacity * 0.9,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`state-borders-health-${fillOpacity}`}
        data={geoData}
        pane="healthPane"
        style={{
          fillOpacity: 0,
          color: "#052e16",
          weight: 0.7,
          opacity: 0.92,
        }}
      />

      {/* ── Organic Wind Sweeping ── */}
      <div style={{
        position: 'absolute', inset: 0,
        zIndex: 395, pointerEvents: 'none', overflow: 'hidden',
        // Two layers of soft diagonal swooshes representing wind over grass/canopy
        backgroundImage: 'repeating-linear-gradient(25deg, transparent, rgba(255,255,255,0.04) 20px, transparent 40px), repeating-linear-gradient(25deg, transparent, rgba(167,243,208,0.06) 60px, transparent 120px)',
        backgroundSize: '200% 200%, 200% 200%',
        animation: 'windFlowMovement 20s linear infinite',
      }} />

      <HealthyFieldsLegend />
    </>
  );
};

const HealthyFieldsLegend = () => (
  <div
    className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 320 }}
  >
    <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">
      Vegetation Vigor & Density
    </div>
    <div className="h-3.5 w-full rounded-full mb-2"
      style={{ background: "linear-gradient(to right, #f5deb3, #d4e157 20%, #9ccc65 40%, #66bb6a 60%, #43a047 80%, #1b5e20)" }} />
    <div className="flex justify-between text-[9px] font-semibold text-white/60">
      <span>Barren</span><span>Sparse</span><span>Normal</span><span>Healthy</span><span>Lush</span>
    </div>
    <div className="text-[8px] text-white/35 text-center mt-2">
      Continuous fluid blending mapping NDVI canopy density
    </div>
  </div>
);
