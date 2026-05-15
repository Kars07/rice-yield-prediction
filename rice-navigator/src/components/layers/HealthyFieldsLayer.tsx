import { useCallback, useEffect, useLayoutEffect, useState, useMemo } from "react";
import { GeoJSON, useMap, Circle, CircleMarker } from "react-leaflet";
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

function pip(lat: number, lng: number, ring: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const hit = yi > lat !== yj > lat && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
    if (hit) inside = !inside;
  }
  return inside;
}

function randomPointInFeature(feature: any): [number, number] | null {
  const geom = feature.geometry;
  if (!geom) return null;
  const rings: number[][][] = [];
  if (geom.type === "Polygon") rings.push(geom.coordinates[0]);
  else if (geom.type === "MultiPolygon")
    geom.coordinates.forEach((p: number[][][]) => rings.push(p[0]));
  if (!rings.length) return null;

  let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
  rings.forEach(ring =>
    ring.forEach(([lng, lat]) => {
      if (lat < minLat) minLat = lat; if (lat > maxLat) maxLat = lat;
      if (lng < minLng) minLng = lng; if (lng > maxLng) maxLng = lng;
    })
  );

  for (let attempt = 0; attempt < 40; attempt++) {
    const lat = minLat + Math.random() * (maxLat - minLat);
    const lng = minLng + Math.random() * (maxLng - minLng);
    if (rings.some(ring => pip(lat, lng, ring))) return [lat, lng];
  }
  return [(minLat + maxLat) / 2, (minLng + maxLng) / 2];
}

// 0 to 1 scale for Health/Yield
const FIELD_STOPS = [
  { t: 0.00, r: 239, g: 68,  b: 68  }, // Red (Critical)
  { t: 0.25, r: 236, g: 72,  b: 153 }, // Pink (Unhealthy)
  { t: 0.50, r: 250, g: 204, b: 21  }, // Yellow (Fair)
  { t: 0.75, r: 132, g: 204, b: 22  }, // Light Green / Yellow-Green (Healthy)
  { t: 1.00, r: 21,  g: 128, b: 61  }, // Thick Green (High Healthy)
];

function fieldToHex(score: number): string {
  const v = Math.min(1, Math.max(0, score));
  for (let i = 0; i < FIELD_STOPS.length - 1; i++) {
    const lo = FIELD_STOPS[i], hi = FIELD_STOPS[i + 1];
    if (v >= lo.t && v <= hi.t) {
      const t = (v - lo.t) / (hi.t - lo.t);
      const r = Math.round(lo.r + (hi.r - lo.r) * t).toString(16).padStart(2, "0");
      const g = Math.round(lo.g + (hi.g - lo.g) * t).toString(16).padStart(2, "0");
      const b = Math.round(lo.b + (hi.b - lo.b) * t).toString(16).padStart(2, "0");
      return `#${r}${g}${b}`;
    }
  }
  return "#22c55e"; // Fallback to standard green instead of forest green
}

function getScore(stateName: string, metrics: Record<string, StateMetrics> | null): number {
  if (!metrics || !metrics[stateName]) return 0.5;
  const ndvi = metrics[stateName].latestNdvi || 0.45; 
  // Strict absolute mapping to ensure average fields are Yellow (0.5) 
  // and only extremely high NDVI hits Thick Green
  if (ndvi <= 0.2) return 0.0;
  if (ndvi >= 0.75) return 1.0;
  return (ndvi - 0.2) / 0.55; 
}

function lgaHealth(lat: number, lng: number, stateScore: number, latMin: number, latMax: number, seed: number): number {
  const latBias = ((lat - latMin) / (latMax - latMin || 1) - 0.5) * 0.1;
  const s = seed * 0.001;
  // Smooth macro variance so the base color reliably reflects the region's overall score
  const v = (
    Math.sin(lat * 15.0 + lng * 6.1 + s) * 0.45 +
    Math.sin(lat * 7.2 - lng * 10.8 + s * 1.5) * 0.35 +
    Math.sin(lat * 22 + s * 2.1) * 0.20
  ) * 0.35; 
  return Math.max(0, Math.min(1, stateScore + latBias + v));
}

function opacityForZoom(z: number): number {
  if (z <= 11) return 0.90;
  return 0.85;
}

export const HealthyFieldsLayer = ({ geoData, metrics }: Props) => {
  const map = useMap();
  const [fillOpacity, setFillOpacity] = useState(0.90);

  useLayoutEffect(() => {
    setFillOpacity(opacityForZoom(map.getZoom()));
  }, [map]);

  // Pane setup + Grainy Field Animation
  useEffect(() => {
    if (!document.getElementById('field-cinematic-style')) {
      const svg = document.createElement('div');
      svg.innerHTML = `
        <svg id="field-texture-svg" width="0" height="0" style="position:absolute;z-index:-1;pointer-events:none;">
          <filter id="fieldTexture">
            <feTurbulence type="fractalNoise" baseFrequency="0.15" numOctaves="3" result="noise">
              <animate attributeName="baseFrequency" values="0.15;0.18;0.15" dur="6s" repeatCount="indefinite" />
            </feTurbulence>
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="15" xChannelSelector="R" yChannelSelector="G" />
          </filter>
        </svg>
      `;
      document.body.appendChild(svg);

      const el = document.createElement('style');
      el.id = 'field-cinematic-style';
      el.textContent = `
        @keyframes fieldBreath {
          0%, 100% { filter: url(#fieldTexture) brightness(1.0) saturate(1.0); opacity: 0.72; }
          50%      { filter: url(#fieldTexture) brightness(1.15) saturate(1.2); opacity: 0.85; }
        }
        @keyframes grainPulse {
          0%, 100% { filter: url(#fieldTexture) blur(6px) brightness(0.9); opacity: 0.9; }
          50%      { filter: url(#fieldTexture) blur(5px) brightness(1.2) saturate(1.1); opacity: 1.0; }
        }
      `;
      document.head.appendChild(el);
    }

    if (!map.getPane('fieldPaneGlow')) map.createPane('fieldPaneGlow');
    const glowPane = map.getPane('fieldPaneGlow')!;
    glowPane.style.zIndex = '340';
    glowPane.style.pointerEvents = 'none';
    glowPane.style.filter = 'blur(25px)'; // Slightly less blur than water to maintain field structure
    (glowPane.style as any).mixBlendMode = 'multiply';
    glowPane.style.opacity = '1.0';

    if (!map.getPane('fieldPaneDots')) map.createPane('fieldPaneDots');
    const dotsPane = map.getPane('fieldPaneDots')!;
    dotsPane.style.zIndex = '345';
    dotsPane.style.pointerEvents = 'none';
    dotsPane.style.filter = 'url(#fieldTexture) blur(6px)'; 
    (dotsPane.style as any).mixBlendMode = 'multiply';
    dotsPane.style.animation = 'grainPulse 3s ease-in-out infinite';
    dotsPane.style.willChange = 'filter, opacity';

    if (!map.getPane('fieldPane')) map.createPane('fieldPane');
    const pane = map.getPane('fieldPane')!;
    pane.style.zIndex = '350';
    pane.style.pointerEvents = 'none';
    (pane.style as any).mixBlendMode = 'multiply';
    pane.style.animation = 'fieldBreath 4s ease-in-out infinite';
    pane.style.willChange = 'filter, opacity';
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
      if (!si) return { ...f, properties: { ...f.properties, fillColor: fieldToHex(0.5) } };
      
      const stateScore = getScore(csvName, metrics);
      const [lat, lng] = centroidOf(f);
      const score = lgaHealth(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
      
      return { ...f, properties: { ...f.properties, fillColor: fieldToHex(score) } };
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
      
      if (height > 0.3) { // Lower threshold to spawn more variance patches for fields
        const s = si.seed + lat;
        const numPatches = Math.min(5, Math.floor(height * 3.5)); // More patches
        
        for (let i = 0; i < numPatches; i++) {
          const pLat = lat + (Math.sin(s + i * 1.5) * height * 0.20);
          const pLng = lng + (Math.cos(s + i * 3.1) * height * 0.20);
          
          const stateScore = getScore(csvName, metrics);
          const fieldVariance = (Math.sin(s + i * 4.7) * 0.70) - 0.20; // Lower bias for pink patches
          const baseScore = lgaHealth(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
          const patchScore = Math.max(0, Math.min(1, baseScore + fieldVariance));
          
          const maxRadiusMeters = height * 111000 * 0.15; 
          const radius = maxRadiusMeters * 0.5 + Math.abs(Math.sin(s + i * 2.3)) * (maxRadiusMeters * 0.5);
          
          const dots = [];
          // Lots of tiny clustered dots to simulate pixelated satellite health mapping
          const numDots = 4 + Math.floor(Math.abs(Math.sin(s + i * 3.1)) * 6);
          for (let d = 0; d < numDots; d++) {
             const dLat = pLat + (Math.sin(s + d * 5.2) * (radius / 111000) * 0.6);
             const dLng = pLng + (Math.cos(s + d * 7.3) * (radius / 111000) * 0.6);
             const dRad = radius * (0.02 + Math.abs(Math.sin(s + d * 2.9)) * 0.06); // Extremely small
             // Heavily bias anomaly dots downward to mimic pink/red crop stress pixels
             const dScore = Math.max(0, Math.min(1, patchScore + (Math.sin(s + d) * 0.9) - 0.25));
             dots.push({ lat: dLat, lng: dLng, score: dScore, radius: dRad });
          }
          
          patches.push({ lat: pLat, lng: pLng, score: patchScore, radius, dots });
        }
      }
    });
    return patches;
  }, [lgaGeoJson, metrics, stateLatRanges]);

  const scatterGrains = useMemo(() => {
    if (!geoData || !metrics) return [];
    const points: { lat: number; lng: number; score: number; radius: number }[] = [];
    
    // Scatter points globally across state features like the Rainfall layer
    geoData.features.forEach((f: any) => {
      const csvName = f.properties?.shapeName ?? "";
      const si = stateLatRanges[csvName];
      if (!si) return;

      const stateScore = getScore(csvName, metrics);
      
      // 1200 dots per state with larger radius to completely carpet the map in multi-color textures
      for (let i = 0; i < 1200; i++) {
        const pt = randomPointInFeature(f);
        if (pt) {
          const [lat, lng] = pt;
          const baseScore = lgaHealth(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
          
          // Massive random variance ensures every region is heavily speckled with pink/red/yellow
          const grainVariance = (Math.random() - 0.5) * 1.5; 
          const score = Math.max(0, Math.min(1, baseScore + grainVariance));
          
          const radius = 3.5 + Math.random() * 4.5; // Large radius 3.5 to 8.0 px
          points.push({ lat, lng, score, radius });
        }
      }
    });
    return points;
  }, [geoData, metrics, stateLatRanges]);

  const lgaGlowStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#22c55e",
    fillOpacity: fillOpacity * 1.1,
    color:       "transparent",
    weight:      0,
  }), [fillOpacity]);

  const lgaStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#22c55e",
    fillOpacity: fillOpacity * 0.20, // Heavily transparent base so the colorful grains stand out
    color:       "#022c22", // Very dark green boundary
    weight:      0.5,
    opacity:     0.72,
  }), [fillOpacity]);

  if (!coloredLgaGeoJson) return null;

  return (
    <>
      <GeoJSON
        key={`field-glow-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="fieldPaneGlow"
        style={lgaGlowStyle}
      />

      {lgaPatches.map((p, i) => (
        <Circle
          key={`patch-glow-${i}-${fillOpacity}`}
          center={[p.lat, p.lng]}
          radius={p.radius}
          pane="fieldPaneGlow"
          pathOptions={{
            fillColor: fieldToHex(p.score),
            fillOpacity: fillOpacity * 1.2,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`field-fill-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="fieldPane"
        style={lgaStyle}
      />

      {lgaPatches.flatMap(p => p.dots).map((d, i) => (
        <Circle
          key={`patch-dot-${i}-${fillOpacity}`}
          center={[d.lat, d.lng]}
          radius={d.radius}
          pane="fieldPaneDots"
          pathOptions={{
            fillColor: fieldToHex(d.score),
            fillOpacity: fillOpacity * 0.95, // Intense dots
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`state-borders-fields-${fillOpacity}`}
        data={geoData}
        pane="fieldPane"
        style={{
          fillOpacity: 0,
          color: "#022c22",
          weight: 0.7,
          opacity: 0.92,
        }}
      />

      {/* ── Granular Scatter Dots (Spreads like grains, exactly like Rainfall Events) ── */}
      {scatterGrains.map((pt, i) => (
        <CircleMarker
          key={`grain-scatter-${i}-${fillOpacity}`}
          center={[pt.lat, pt.lng]}
          radius={pt.radius}
          pane="fieldPaneDots"
          pathOptions={{
            fillColor: fieldToHex(pt.score),
            fillOpacity: fillOpacity * 0.95,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      
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
      Yield & Health Predictor · LGA Regions
    </div>
    <div className="h-3.5 w-full rounded-full mb-2"
      style={{ background: "linear-gradient(to right, #ef4444, #ec4899 25%, #facc15 50%, #84cc16 75%, #15803d)" }} />
    <div className="flex justify-between text-[9px] font-semibold text-white/60">
      <span>Critical</span><span>Unhealthy</span><span>Fair</span><span>Healthy</span><span>High Healthy</span>
    </div>
    <div className="text-[8px] text-white/35 text-center mt-2">
      Fluid satellite vegetation anomaly mapping
    </div>
  </div>
);
