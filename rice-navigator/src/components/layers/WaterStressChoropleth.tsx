import { useCallback, useEffect, useLayoutEffect, useState, useMemo } from "react";
import { GeoJSON, useMap, Circle, Polyline, Tooltip } from "react-leaflet";
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

// 0 to 1 scale. 0 = Dry Beige, 1 = Navy Flooded
const WATER_STOPS = [
  { t: 0.00, r: 245, g: 235, b: 200 }, // Dry (pale beige)
  { t: 0.20, r: 180, g: 240, b: 240 }, // Mild (pale cyan)
  { t: 0.40, r: 0,   g: 220, b: 255 }, // Normal (cyan)
  { t: 0.60, r: 0,   g: 140, b: 255 }, // Wet (vivid blue)
  { t: 0.80, r: 0,   g: 60,  b: 220 }, // High wet (deep blue)
  { t: 1.00, r: 0,   g: 0,   b: 140 }, // Flooded (navy)
];

function waterToHex(score: number): string {
  const v = Math.min(1, Math.max(0, score));
  for (let i = 0; i < WATER_STOPS.length - 1; i++) {
    const lo = WATER_STOPS[i], hi = WATER_STOPS[i + 1];
    if (v >= lo.t && v <= hi.t) {
      const t = (v - lo.t) / (hi.t - lo.t);
      const r = Math.round(lo.r + (hi.r - lo.r) * t).toString(16).padStart(2, "0");
      const g = Math.round(lo.g + (hi.g - lo.g) * t).toString(16).padStart(2, "0");
      const b = Math.round(lo.b + (hi.b - lo.b) * t).toString(16).padStart(2, "0");
      return `#${r}${g}${b}`;
    }
  }
  return "#00008c";
}

function getScore(stateName: string, metrics: Record<string, StateMetrics> | null): number {
  if (!metrics || !metrics[stateName]) return 0.4;
  const ndwi = metrics[stateName].latestNdwi;   // e.g. -0.5 to 0.1
  const precip = metrics[stateName].avgPrecipitation; // mm
  const ndwiNorm = Math.min(1, Math.max(0, (ndwi + 0.5) / 0.6));
  const precipNorm = Math.min(1, Math.max(0, (precip - 1) / 9));
  return (ndwiNorm * 0.5 + precipNorm * 0.5);
}

function lgaWater(lat: number, lng: number, stateScore: number, latMin: number, latMax: number, seed: number): number {
  const latBias = ((latMin - lat) / (latMax - latMin || 1)) * 0.15; // Usually wetter in the south
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

export const WaterStressChoropleth = ({ geoData, metrics }: Props) => {
  const map = useMap();
  const [fillOpacity, setFillOpacity] = useState(0.90);

  useLayoutEffect(() => {
    setFillOpacity(opacityForZoom(map.getZoom()));
  }, [map]);

  // Pane setup + water flowing animation
  useEffect(() => {
    if (!document.getElementById('water-cinematic-style')) {
      const svg = document.createElement('div');
      svg.innerHTML = `
        <svg id="water-flow-svg" width="0" height="0" style="position:absolute;z-index:-1;pointer-events:none;">
          <filter id="waterFlowFilter">
            <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves="3" result="noise">
              <animate attributeName="baseFrequency" values="0.015;0.022;0.015" dur="8s" repeatCount="indefinite" />
            </feTurbulence>
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="5" xChannelSelector="R" yChannelSelector="B" />
          </filter>
        </svg>
      `;
      document.body.appendChild(svg);

      const el = document.createElement('style');
      el.id = 'water-cinematic-style';
      el.textContent = `
        @keyframes waterBreath {
          0%, 100% { filter: url(#waterFlowFilter) brightness(1.0) saturate(1.0); }
          50%      { filter: url(#waterFlowFilter) brightness(1.1) saturate(1.2); }
        }
      `;
      document.head.appendChild(el);
    }

    if (!map.getPane('waterPaneGlow')) map.createPane('waterPaneGlow');
    const glowPane = map.getPane('waterPaneGlow')!;
    glowPane.style.zIndex = '340';
    glowPane.style.pointerEvents = 'none';
    glowPane.style.filter = 'blur(35px)';
    (glowPane.style as any).mixBlendMode = 'multiply';
    glowPane.style.opacity = '1.0';

    if (!map.getPane('waterPaneDots')) map.createPane('waterPaneDots');
    const dotsPane = map.getPane('waterPaneDots')!;
    dotsPane.style.zIndex = '345';
    dotsPane.style.pointerEvents = 'none';
    dotsPane.style.filter = 'blur(8px)';
    (dotsPane.style as any).mixBlendMode = 'multiply';
    dotsPane.style.opacity = '0.9';

    if (!map.getPane('waterPane')) map.createPane('waterPane');
    const pane = map.getPane('waterPane')!;
    pane.style.zIndex = '350';
    pane.style.pointerEvents = 'none';
    (pane.style as any).mixBlendMode = 'multiply';
    pane.style.animation = 'waterBreath 6s ease-in-out infinite';
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
      if (!si) return { ...f, properties: { ...f.properties, fillColor: waterToHex(0.4) } };
      
      const stateScore = getScore(csvName, metrics);
      const [lat, lng] = centroidOf(f);
      const score = lgaWater(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
      
      return { ...f, properties: { ...f.properties, fillColor: waterToHex(score) } };
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
          const waterVariance = Math.sin(s + i * 4.1) * 0.25; 
          const baseScore = lgaWater(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
          const patchScore = Math.max(0, Math.min(1, baseScore + waterVariance));
          
          const maxRadiusMeters = height * 111000 * 0.20; 
          const radius = maxRadiusMeters * 0.6 + Math.abs(Math.sin(s + i * 1.7)) * (maxRadiusMeters * 0.4);
          
          const dots = [];
          const numDots = 2 + Math.floor(Math.abs(Math.sin(s + i * 3.1)) * 3);
          for (let d = 0; d < numDots; d++) {
             const dLat = pLat + (Math.sin(s + d * 5.2) * (radius / 111000) * 0.4);
             const dLng = pLng + (Math.cos(s + d * 7.3) * (radius / 111000) * 0.4);
             const dRad = radius * (0.05 + Math.abs(Math.sin(s + d * 2.9)) * 0.10);
             // Shift score up for intense deep water core
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
    fillColor:   feature?.properties?.fillColor ?? "#3b82f6",
    fillOpacity: fillOpacity * 1.1,
    color:       "transparent",
    weight:      0,
  }), [fillOpacity]);

  const lgaStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#3b82f6",
    fillOpacity: fillOpacity * 0.65,
    color:       "#0f172a", // Dark border for crisp water sections
    weight:      0.5,
    opacity:     0.72,
  }), [fillOpacity]);

  if (!coloredLgaGeoJson) return null;

  return (
    <>
      <GeoJSON
        key={`water-glow-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="waterPaneGlow"
        style={lgaGlowStyle}
      />

      {lgaPatches.map((p, i) => (
        <Circle
          key={`patch-glow-${i}-${fillOpacity}`}
          center={[p.lat, p.lng]}
          radius={p.radius}
          pane="waterPaneGlow"
          pathOptions={{
            fillColor: waterToHex(p.score),
            fillOpacity: fillOpacity * 1.2,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`water-fill-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="waterPane"
        style={lgaStyle}
      />

      {lgaPatches.flatMap(p => p.dots).map((d, i) => (
        <Circle
          key={`patch-dot-${i}-${fillOpacity}`}
          center={[d.lat, d.lng]}
          radius={d.radius}
          pane="waterPaneDots"
          pathOptions={{
            fillColor: waterToHex(d.score),
            fillOpacity: fillOpacity * 0.9,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`state-borders-water-${fillOpacity}`}
        data={geoData}
        pane="waterPane"
        style={{
          fillOpacity: 0,
          color: "#0f172a",
          weight: 0.7,
          opacity: 0.92,
        }}
      />

      <RiverOverlays />
      <WaterStressLegend />
    </>
  );
};

const RiverOverlays = () => {
  const nigerRiver: [number, number][] = [
    [13.5, 3.9], [12.0, 4.2], [10.0, 4.5],
    [8.5, 5.2], [7.8, 6.5], [7.2, 6.7],
    [6.4, 6.4], [5.5, 6.2],
  ];
  const benueRiver: [number, number][] = [
    [8.3, 13.5], [7.9, 11.5], [7.8, 9.8],
    [7.7, 8.3], [7.5, 7.0], [7.2, 6.7],
  ];

  return (
    <>
      <Polyline positions={nigerRiver} pathOptions={{ color: "#38bdf8", weight: 3, opacity: 0.9 }}>
        <Tooltip permanent direction="top" offset={[0, -4]} className="!bg-transparent !border-0 !shadow-none !text-[#0f172a] !font-bold !text-[10px]">
          Niger River
        </Tooltip>
      </Polyline>
      <Polyline positions={benueRiver} pathOptions={{ color: "#38bdf8", weight: 2.5, opacity: 0.9 }}>
        <Tooltip permanent direction="top" offset={[0, -4]} className="!bg-transparent !border-0 !shadow-none !text-[#0f172a] !font-bold !text-[10px]">
          Benue River
        </Tooltip>
      </Polyline>
    </>
  );
};

const WaterStressLegend = () => (
  <div
    className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 320 }}
  >
    <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">
      Water Flow Intensity · LGA Regions
    </div>
    <div className="h-3.5 w-full rounded-full mb-2"
      style={{ background: "linear-gradient(to right, #f5ebc8, #b4f0f0 20%, #00dcff 40%, #008cff 60%, #003cdc 80%, #00008c)" }} />
    <div className="flex justify-between text-[9px] font-semibold text-white/60">
      <span>Dry</span><span>Mild</span><span>Normal</span><span>Wet</span><span>Flooded</span>
    </div>
    <div className="text-[8px] text-white/35 text-center mt-2">
      Continuous fluid blending mapping NDWI & Precipitation
    </div>
  </div>
);
