import { useCallback, useEffect, useLayoutEffect, useRef, useState, useMemo } from "react";
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

function lgaTemp(lat: number, lng: number, stateAvg: number, latMin: number, latMax: number, seed: number): number {
  const latBias = ((lat - latMin) / (latMax - latMin || 1) - 0.5) * 5.0;
  const s = seed * 0.001;
  const v = (
    Math.sin(lat * 8.0 + lng * 3.1 + s) * 0.42 +
    Math.sin(lat * 3.2 - lng * 6.8 + s * 1.5) * 0.35 +
    Math.sin(lat * 13 + s * 2.1) * 0.15 +
    Math.sin(lng * 9.4 + s * 0.7) * 0.08
  ) * 4.8;
  return Math.max(20, Math.min(34, stateAvg + latBias + v));
}

// Multi-hue scientific ramp — matches real GIS lava/thermal probability maps
// pale yellow-green → bright yellow → vivid orange → orange-red → bright red → deep red → dark maroon
const STOPS = [
  { t: 0.00, r: 222, g: 235, b: 140 },  // 24°C — pale yellow-green (cool terrain)
  { t: 0.18, r: 248, g: 218, b: 20 },  // 25.4°C — bright scientific yellow
  { t: 0.34, r: 252, g: 150, b: 0 },  // 26.7°C — vivid amber-orange
  { t: 0.50, r: 235, g: 68, b: 0 },  // 28°C — hot orange-red
  { t: 0.65, r: 196, g: 18, b: 0 },  // 29.2°C — bright volcanic red
  { t: 0.80, r: 130, g: 5, b: 0 },  // 30.4°C — deep rich red
  { t: 1.00, r: 52, g: 0, b: 0 },  // 32°C+ — near-black maroon (hottest)
];
function tempToHex(temp: number): string {
  const v = Math.min(1, Math.max(0, (temp - 20) / 14)); // 20–34°C → full palette range
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

// Keep colors vivid at all zoom levels — multiply blend mode lets map details show through
function opacityForZoom(z: number): number {
  if (z <= 11) return 0.90;
  return 0.78;  // only slight fade at extreme close-up zoom
}

export const HeatSignatureLayer = ({ geoData, metrics }: Props) => {
  const map = useMap();

  // fillOpacity state — changing key forces layer recreation with new opacity
  const [fillOpacity, setFillOpacity] = useState(0.90);
  useLayoutEffect(() => {
    setFillOpacity(opacityForZoom(map.getZoom()));
  }, [map]);

  // Pane setup + cinematic animation injection + fluid spread effect
  useEffect(() => {
    // Inject SVG Heat Haze distortion filter + CSS keyframes
    if (!document.getElementById('heat-cinematic-style')) {
      const svg = document.createElement('div');
      svg.innerHTML = `
        <svg id="heat-haze-svg" width="0" height="0" style="position:absolute;z-index:-1;pointer-events:none;">
          <filter id="heatHaze">
            <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves="2" result="noise">
              <animate attributeName="baseFrequency" values="0.015;0.025;0.015" dur="6s" repeatCount="indefinite" />
            </feTurbulence>
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="4" xChannelSelector="R" yChannelSelector="G" />
          </filter>
        </svg>
      `;
      document.body.appendChild(svg);

      const el = document.createElement('style');
      el.id = 'heat-cinematic-style';
      el.textContent = `
        @keyframes heatBreath {
          0%, 100% { filter: url(#heatHaze) brightness(1.0)  saturate(1.0);  }
          50%       { filter: url(#heatHaze) brightness(1.12) saturate(1.25); }
        }
        @keyframes heatShimmer {
          0%   { background-position: -250% 0; opacity: 0;  }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { background-position: 350% 0;  opacity: 0;  }
        }
        @keyframes heatGlowPulse {
          0%, 100% { opacity: 0.15; transform: scale(1.0);   }
          50%       { opacity: 0.60; transform: scale(1.06); }
        }
        @keyframes heatEdgePulse {
          0%, 100% { opacity: 0.0; }
          50%       { opacity: 0.35; }
        }
      `;
      document.head.appendChild(el);
    }

    // 1. Fluid Glow Pane (rendered behind, heavily blurred)
    if (!map.getPane('heatPaneGlow')) map.createPane('heatPaneGlow');
    const glowPane = map.getPane('heatPaneGlow')!;
    glowPane.style.zIndex = '340';
    glowPane.style.pointerEvents = 'none';
    glowPane.style.filter = 'blur(35px)'; // Heavier blur for smoother liquid spread
    (glowPane.style as any).mixBlendMode = 'multiply';
    glowPane.style.opacity = '1.0'; // Max opacity for thickest fluid base

    // 2. Main Heat Pane (sharp, multiply blend)
    if (!map.getPane('heatPane')) map.createPane('heatPane');
    const pane = map.getPane('heatPane')!;
    pane.style.zIndex = '350';
    pane.style.pointerEvents = 'none';
    (pane.style as any).mixBlendMode = 'multiply';
    pane.style.animation = 'heatBreath 4s ease-in-out infinite';
    pane.style.willChange = 'filter';
  }, [map]);

  // Zoom listener — updates fillOpacity state → triggers re-render → react-leaflet
  // detects style prop changed → calls instance.setStyle() on all 159 LGA paths
  useEffect(() => {
    const onZoomEnd = () => setFillOpacity(opacityForZoom(map.getZoom()));
    map.on("zoomend", onZoomEnd);
    return () => { map.off("zoomend", onZoomEnd); };
  }, [map]);

  // Fetch LGA GeoJSON once
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
      const m = metrics[csvName];
      const si = stateLatRanges[csvName];
      if (!m || !si) {
        return { ...f, properties: { ...f.properties, fillColor: tempToHex(28.0) } };
      }
      const [lat, lng] = centroidOf(f);
      const temp = lgaTemp(lat, lng, m.avgTemperature, si.latMin, si.latMax, si.seed);
      return { ...f, properties: { ...f.properties, fillColor: tempToHex(temp) } };
    });
    return { type: "FeatureCollection" as const, features };
  }, [lgaGeoJson, metrics, stateLatRanges]);

  // Generate localized temperature variance patches for very large LGAs
  const lgaPatches = useMemo(() => {
    if (!lgaGeoJson || !metrics) return [];
    const patches: { lat: number; lng: number; temp: number; radius: number }[] = [];
    
    lgaGeoJson.features.forEach((f: any) => {
      const csvName = STATE_NAME_MAP[f.properties?.NAME_1] ?? "";
      const m = metrics[csvName];
      const si = stateLatRanges[csvName];
      if (!m || !si) return;

      const [lat, lng] = centroidOf(f);
      const [latMin, latMax] = bboxLatRange(f);
      const height = latMax - latMin;
      
      // Only target physically huge regions (e.g. > 0.45 degrees approx 50km)
      if (height > 0.45) {
        const s = si.seed + lat; // pseudo-random seed based on position
        const numPatches = Math.min(3, Math.floor(height * 2.5)); // 1 to 3 distinct sub-patches
        
        for (let i = 0; i < numPatches; i++) {
          const pLat = lat + (Math.sin(s + i * 1.3) * height * 0.35);
          const pLng = lng + (Math.cos(s + i * 2.7) * height * 0.35);
          
          // Introduce a +/- 2.2 degree local variance to break up uniform colors softly
          const tempVariance = Math.sin(s + i * 4.1) * 2.2; 
          const baseTemp = lgaTemp(lat, lng, m.avgTemperature, si.latMin, si.latMax, si.seed);
          const patchTemp = Math.max(20, Math.min(34, baseTemp + tempVariance));
          
          // Much larger patches so they bleed as big waves, not dots (10km to 28km radius)
          const radius = 10000 + Math.abs(Math.sin(s + i * 1.7)) * 18000;
          patches.push({ lat: pLat, lng: pLng, temp: patchTemp, radius });
        }
      }
    });
    return patches;
  }, [lgaGeoJson, metrics, stateLatRanges]);

  const lgaGlowStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#ff6400",
    fillOpacity: fillOpacity * 1.1, // Oversaturate the fluid base for thick liquid mass
    color:       "transparent",
    weight:      0,
  }), [fillOpacity]);

  const lgaStyle = useCallback((feature: any) => ({
    fillColor:   feature?.properties?.fillColor ?? "#ff6400",
    fillOpacity: fillOpacity * 0.65, // Increase sharp opacity to anchor the thick colors
    color:       "#111",
    weight:      0.5,
    opacity:     0.72,
  }), [fillOpacity]);

  if (!coloredLgaGeoJson) return null;

  return (
    <>
      {/* ── Fluid Spread Layer (Blurred) ── */}
      <GeoJSON
        key={`lga-heat-glow-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="heatPaneGlow"
        style={lgaGlowStyle}
      />

      {/* ── Intra-region variance patches (Fluid Glow) ── */}
      {lgaPatches.map((p, i) => (
        <Circle
          key={`patch-glow-${i}-${fillOpacity}`}
          center={[p.lat, p.lng]}
          radius={p.radius * 2.0} // Massive radius to ensure they overlap and bleed seamlessly
          pane="heatPaneGlow"
          pathOptions={{
            fillColor: tempToHex(p.temp),
            fillOpacity: fillOpacity * 1.2,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      {/* ── Sharp Details Layer ── */}
      <GeoJSON
        key={`lga-heat-fill-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="heatPane"
        style={lgaStyle}
      />

      {/* Removed the sharp intra-region patches entirely — we only want them in the blurred glow pane */}

      {/* State outer borders — weight matches old inner LGA line (~1.5px) */}
      <GeoJSON
        key={`state-borders-${fillOpacity}`}
        data={geoData}
        pane="heatPane"
        style={{
          fillOpacity: 0,
          color: "#111",
          weight: 0.7,
          opacity: 0.92,
        }}
      />

      {/* ── Cinematic shimmer sweep ── diagonal golden shine ─────────── */}
      <div style={{
        position: 'absolute', inset: 0,
        zIndex: 395, pointerEvents: 'none', overflow: 'hidden',
        backgroundImage: 'linear-gradient(108deg, transparent, rgba(255,210,80,0.18) 40%, rgba(255,240,160,0.10) 50%, rgba(255,210,80,0.18) 60%, transparent)',
        backgroundSize: '300% 100%',
        animation: 'heatShimmer 9s ease-in-out infinite',
      }} />

      {/* ── Radial amber glow ── warm pulse from map center ───────────── */}
      <div style={{
        position: 'absolute', inset: 0,
        zIndex: 394, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 55% 40% at 50% 50%, rgba(255,140,20,0.10) 0%, rgba(200,80,0,0.03) 55%, transparent 100%)',
        animation: 'heatGlowPulse 5s ease-in-out infinite',
        transformOrigin: 'center',
      }} />

      {/* ── Edge heat vignette ── breathes at perimeter ───────────────── */}
      <div style={{
        position: 'absolute', inset: 0,
        zIndex: 393, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 100% 100% at 50% 50%, transparent 45%, rgba(120,15,0,0.14) 100%)',
        animation: 'heatEdgePulse 6s ease-in-out infinite',
        animationDelay: '1.5s',
      }} />

      {/* Legend */}
      <div
        className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
        style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 320 }}
      >
        <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">
          Surface Temperature · LGA Regions
        </div>
        <div className="h-3.5 w-full rounded-full mb-2"
          style={{ background: "linear-gradient(to right, #deeb8c, #f8da14 18%, #fc9600 34%, #eb4400 50%, #c41200 65%, #820500 80%, #340000)" }} />
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
