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

// 0.0: Arid/Single (Bright Yellow) -> 0.2: Arid/Multi (Orange)
// 0.4: Dry/Single (Lime) -> 0.6: Dry/Multi (Olive)
// 0.8: No Dry/Single (Cyan) -> 1.0: No Dry/Multi (Royal Blue)
const IRRIGATION_STOPS = [
  { t: 0.00, r: 253, g: 224, b: 71  }, // Bright Yellow (#fde047)
  { t: 0.20, r: 249, g: 115, b: 22  }, // Orange (#f97316)
  { t: 0.40, r: 163, g: 230, b: 53  }, // Light Lime Green (#a3e635)
  { t: 0.60, r: 77,  g: 124, b: 15  }, // Deep Olive Green (#4d7c0f)
  { t: 0.80, r: 34,  g: 211, b: 238 }, // Cyan / Light Blue (#22d3ee)
  { t: 1.00, r: 29,  g: 78,  b: 216 }, // Dark Royal Blue (#1d4ed8)
];

function irrigationToHex(score: number): string {
  const v = Math.min(1, Math.max(0, score));
  for (let i = 0; i < IRRIGATION_STOPS.length - 1; i++) {
    const lo = IRRIGATION_STOPS[i], hi = IRRIGATION_STOPS[i + 1];
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
  
  let climateScore = 0; // Arid (< 60mm avg precip)
  if (m.avgPrecipitation >= 100) climateScore = 0.8; // No dry season
  else if (m.avgPrecipitation >= 60) climateScore = 0.4; // Dry season
  
  // If moisture deficit (NDWI) is relatively low during dry periods, it implies irrigation (Multi-crop)
  const isMultiCrop = m.latestNdwi > -0.32; 
  
  if (climateScore === 0) return isMultiCrop ? 0.2 : 0.0;
  if (climateScore === 0.4) return isMultiCrop ? 0.6 : 0.4;
  return isMultiCrop ? 1.0 : 0.8;
}

function lgaIrrigation(lat: number, lng: number, stateScore: number, latMin: number, latMax: number, seed: number): number {
  const s = seed * 0.001;
  const v = (
    Math.sin(lat * 8.0 + lng * 3.1 + s) * 0.42 +
    Math.sin(lat * 3.2 - lng * 6.8 + s * 1.5) * 0.35 +
    Math.sin(lat * 13 + s * 2.1) * 0.15 +
    Math.sin(lng * 9.4 + s * 0.7) * 0.08
  ) * 0.15; // ±0.15 variance to bleed between MRU sub-zones nicely
  return Math.max(0, Math.min(1, stateScore + v));
}

function opacityForZoom(z: number): number {
  if (z <= 11) return 0.85;
  return 0.75;
}

export const IrrigationZonesChoropleth = ({ geoData, metrics }: Props) => {
  const map = useMap();
  const [fillOpacity, setFillOpacity] = useState(0.85);

  useLayoutEffect(() => {
    setFillOpacity(opacityForZoom(map.getZoom()));
  }, [map]);

  // Pane setup + organic pulsating animation
  useEffect(() => {
    if (!document.getElementById('irrigation-cinematic-style')) {
      const svg = document.createElement('div');
      svg.innerHTML = `
        <svg id="confluence-flow-svg" width="0" height="0" style="position:absolute;z-index:-1;pointer-events:none;">
          <filter id="confluenceWaterFilter">
            <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="4" result="noise">
              <animate attributeName="baseFrequency" values="0.02;0.028;0.02" dur="6s" repeatCount="indefinite" />
            </feTurbulence>
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="8" xChannelSelector="R" yChannelSelector="B" />
          </filter>
        </svg>
      `;
      document.body.appendChild(svg);

      const el = document.createElement('style');
      el.id = 'irrigation-cinematic-style';
      el.textContent = `
        @keyframes irrigationBreath {
          0%, 100% { filter: brightness(1.0) saturate(1.0); }
          50%      { filter: brightness(1.15) saturate(1.2); }
        }
        @keyframes irrigationFlowMovement {
          0%   { background-position: 0px 0px, 0px 0px; }
          100% { background-position: -400px 400px, 800px 400px; }
        }
        @keyframes waterFlowDash {
          to { stroke-dashoffset: -22; }
        }
        @keyframes confluencePulse {
          0%, 100% { transform: scale(1.0); opacity: 0.8; }
          50%      { transform: scale(1.12); opacity: 0.95; filter: drop-shadow(0 0 10px rgba(6,182,212,0.8)); }
        }
        .confluence-flow-line {
          animation: waterFlowDash 1s linear infinite;
          stroke-linecap: round;
          filter: drop-shadow(0 0 4px rgba(34,211,238,0.7));
        }
        .confluence-glow-base {
          stroke-linecap: round;
          filter: blur(2px) opacity(0.8);
        }
        .confluence-pulsing-core {
          transform-origin: center;
          animation: confluencePulse 2.5s ease-in-out infinite;
          filter: url(#confluenceWaterFilter);
        }
      `;
      document.head.appendChild(el);
    }

    if (!map.getPane('irrigPaneGlow')) map.createPane('irrigPaneGlow');
    const glowPane = map.getPane('irrigPaneGlow')!;
    glowPane.style.zIndex = '340';
    glowPane.style.pointerEvents = 'none';
    glowPane.style.filter = 'blur(35px)';
    (glowPane.style as any).mixBlendMode = 'multiply';
    glowPane.style.opacity = '1.0';

    if (!map.getPane('irrigPaneDots')) map.createPane('irrigPaneDots');
    const dotsPane = map.getPane('irrigPaneDots')!;
    dotsPane.style.zIndex = '345';
    dotsPane.style.pointerEvents = 'none';
    dotsPane.style.filter = 'blur(8px)';
    (dotsPane.style as any).mixBlendMode = 'multiply';
    dotsPane.style.opacity = '0.9';

    if (!map.getPane('irrigPane')) map.createPane('irrigPane');
    const pane = map.getPane('irrigPane')!;
    pane.style.zIndex = '350';
    pane.style.pointerEvents = 'none';
    (pane.style as any).mixBlendMode = 'multiply';
    pane.style.animation = 'irrigationBreath 7s ease-in-out infinite';
    pane.style.willChange = 'filter';

    if (!map.getPane('confluencePane')) map.createPane('confluencePane');
    const confluencePane = map.getPane('confluencePane')!;
    confluencePane.style.zIndex = '360';
    confluencePane.style.pointerEvents = 'auto';
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
      if (!si) return { ...f, properties: { ...f.properties, fillColor: irrigationToHex(0.0) } };
      
      const stateScore = getScore(csvName, metrics);
      const [lat, lng] = centroidOf(f);
      const score = lgaIrrigation(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
      
      return { ...f, properties: { ...f.properties, fillColor: irrigationToHex(score) } };
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
          const varOffset = Math.sin(s + i * 4.1) * 0.15; 
          const baseScore = lgaIrrigation(lat, lng, stateScore, si.latMin, si.latMax, si.seed);
          const patchScore = Math.max(0, Math.min(1, baseScore + varOffset));
          
          const maxRadiusMeters = height * 111000 * 0.20; 
          const radius = maxRadiusMeters * 0.6 + Math.abs(Math.sin(s + i * 1.7)) * (maxRadiusMeters * 0.4);
          
          const dots = [];
          const numDots = 2 + Math.floor(Math.abs(Math.sin(s + i * 3.1)) * 3);
          for (let d = 0; d < numDots; d++) {
             const dLat = pLat + (Math.sin(s + d * 5.2) * (radius / 111000) * 0.4);
             const dLng = pLng + (Math.cos(s + d * 7.3) * (radius / 111000) * 0.4);
             const dRad = radius * (0.05 + Math.abs(Math.sin(s + d * 2.9)) * 0.10);
             // Dots shift score up heavily for intense localized multi-crop hotspots
             const dScore = Math.min(1, patchScore + 0.1 + Math.sin(s + d) * 0.1);
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
    fillOpacity: fillOpacity * 0.7,
    color:       "#3f3f46", // Dark gray borders
    weight:      0.6,
    opacity:     0.5,
  }), [fillOpacity]);

  if (!coloredLgaGeoJson) return null;

  return (
    <>
      <GeoJSON
        key={`irrig-glow-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="irrigPaneGlow"
        style={lgaGlowStyle}
      />

      {lgaPatches.map((p, i) => (
        <Circle
          key={`patch-glow-${i}-${fillOpacity}`}
          center={[p.lat, p.lng]}
          radius={p.radius}
          pane="irrigPaneGlow"
          pathOptions={{
            fillColor: irrigationToHex(p.score),
            fillOpacity: fillOpacity * 1.2,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`irrig-fill-${fillOpacity}`}
        data={coloredLgaGeoJson}
        pane="irrigPane"
        style={lgaStyle}
      />

      {lgaPatches.flatMap(p => p.dots).map((d, i) => (
        <Circle
          key={`patch-dot-${i}-${fillOpacity}`}
          center={[d.lat, d.lng]}
          radius={d.radius}
          pane="irrigPaneDots"
          pathOptions={{
            fillColor: irrigationToHex(d.score),
            fillOpacity: fillOpacity * 0.9,
            color: "transparent",
            weight: 0
          }}
        />
      ))}

      <GeoJSON
        key={`state-borders-irrig-${fillOpacity}`}
        data={geoData}
        pane="irrigPane"
        style={{
          fillOpacity: 0,
          color: "#27272a",
          weight: 0.8,
          opacity: 0.8,
        }}
      />

      {/* ── Subtly sweeping overlays to simulate ecological zones ── */}
      <div style={{
        position: 'absolute', inset: 0,
        zIndex: 395, pointerEvents: 'none', overflow: 'hidden',
        backgroundImage: 'repeating-linear-gradient(45deg, transparent, rgba(255,255,255,0.02) 40px, transparent 80px), repeating-linear-gradient(-45deg, transparent, rgba(0,0,0,0.03) 30px, transparent 60px)',
        backgroundSize: '200% 200%, 200% 200%',
        animation: 'irrigationFlowMovement 35s linear infinite',
      }} />

      <ConfluenceOverlays />
      <IrrigationLegend />
    </>
  );
};

const ConfluenceOverlays = () => {
  const confluences = [
    // --- KANO STATE ---
    {
      name: "Tamburawa Confluence",
      lga: "Dawakin Kudu LGA",
      state: "Kano",
      center: [11.87, 8.53] as [number, number],
      rivers: [
        [[12.02, 8.48], [11.87, 8.53]] as [number, number][],
        [[11.92, 8.35], [11.87, 8.53]] as [number, number][],
        [[11.87, 8.53], [11.78, 8.72]] as [number, number][],
      ],
    },
    {
      name: "Watari-Kano River Junction",
      lga: "Bagwai LGA",
      state: "Kano",
      center: [12.15, 8.24] as [number, number],
      rivers: [
        [[12.28, 8.15], [12.15, 8.24]] as [number, number][],
        [[12.18, 8.38], [12.15, 8.24], [12.05, 8.31]] as [number, number][],
      ],
    },
    // --- KEBBI STATE ---
    {
      name: "Niger-Sokoto Confluence",
      lga: "Bagudo LGA",
      state: "Kebbi",
      center: [11.41, 4.15] as [number, number],
      rivers: [
        [[11.62, 4.02], [11.41, 4.15], [11.18, 4.22]] as [number, number][],
        [[11.68, 4.35], [11.41, 4.15]] as [number, number][],
      ],
    },
    {
      name: "Sokoto-Rima Confluence",
      lga: "Argungu LGA",
      state: "Kebbi",
      center: [12.45, 4.25] as [number, number],
      rivers: [
        [[12.65, 4.12], [12.45, 4.25]] as [number, number][],
        [[12.35, 4.45], [12.45, 4.25], [12.22, 4.18]] as [number, number][],
      ],
    },
    {
      name: "Ka-Sokoto Confluence",
      lga: "Bunza LGA",
      state: "Kebbi",
      center: [11.75, 3.98] as [number, number],
      rivers: [
        [[11.90, 4.12], [11.75, 3.98]] as [number, number][],
        [[11.72, 3.82], [11.75, 3.98], [11.60, 3.92]] as [number, number][],
      ],
    },
    // --- NIGER STATE ---
    {
      name: "Mureji Confluence",
      lga: "Mokwa LGA",
      state: "Niger",
      center: [8.75, 5.9] as [number, number],
      rivers: [
        [[9.02, 5.02], [8.75, 5.9], [8.48, 6.22]] as [number, number][],
        [[9.22, 6.02], [8.75, 5.9]] as [number, number][],
      ],
    },
    {
      name: "Gurara-Niger Confluence",
      lga: "Lapai LGA",
      state: "Niger",
      center: [8.18, 6.67] as [number, number],
      rivers: [
        [[8.42, 6.82], [8.18, 6.67]] as [number, number][],
        [[8.22, 6.45], [8.18, 6.67], [8.05, 6.78]] as [number, number][],
      ],
    },
    {
      name: "Gbako-Niger Confluence",
      lga: "Katcha LGA",
      state: "Niger",
      center: [8.65, 6.05] as [number, number],
      rivers: [
        [[8.88, 6.12], [8.65, 6.05]] as [number, number][],
        [[8.68, 5.88], [8.65, 6.05], [8.52, 6.18]] as [number, number][],
      ],
    },
    // --- JIGAWA STATE ---
    {
      name: "Hadejia-Jama'are Wetlands",
      lga: "Hadejia LGA",
      state: "Jigawa",
      center: [12.65, 10.45] as [number, number],
      rivers: [
        [[12.52, 10.12], [12.65, 10.45]] as [number, number][],
        [[12.42, 10.38], [12.65, 10.45]] as [number, number][],
        [[12.65, 10.45], [12.78, 10.82]] as [number, number][],
      ],
    },
    {
      name: "Iggi-Hadejia Junction",
      lga: "Auyo LGA",
      state: "Jigawa",
      center: [12.52, 9.95] as [number, number],
      rivers: [
        [[12.62, 9.82], [12.52, 9.95]] as [number, number][],
        [[12.42, 9.92], [12.52, 9.95], [12.48, 10.15]] as [number, number][],
      ],
    },
    // --- EBONYI STATE ---
    {
      name: "Cross River-Abonyi Confluence",
      lga: "Afikpo LGA",
      state: "Ebonyi",
      center: [6.05, 8.12] as [number, number],
      rivers: [
        [[6.28, 7.98], [6.05, 8.12]] as [number, number][],
        [[6.18, 8.28], [6.05, 8.12], [5.78, 7.98]] as [number, number][],
      ],
    },
    // --- TARABA STATE ---
    {
      name: "Taraba-Benue Confluence",
      lga: "Gassol LGA",
      state: "Taraba",
      center: [8.64, 10.25] as [number, number],
      rivers: [
        [[8.92, 9.78], [8.64, 10.25], [8.38, 10.82]] as [number, number][],
        [[8.18, 10.12], [8.64, 10.25]] as [number, number][],
      ],
    },
    {
      name: "Benue-Donga Confluence",
      lga: "Wukari LGA",
      state: "Taraba",
      center: [7.95, 10.42] as [number, number],
      rivers: [
        [[8.15, 10.18], [7.95, 10.42], [7.78, 10.65]] as [number, number][],
        [[7.82, 10.22], [7.95, 10.42]] as [number, number][],
      ],
    },
    {
      name: "Benue-Katsina Ala Confluence",
      lga: "Katsina-Ala border LGA",
      state: "Taraba",
      center: [7.80, 9.45] as [number, number],
      rivers: [
        [[8.02, 9.28], [7.80, 9.45], [7.65, 9.68]] as [number, number][],
        [[7.68, 9.32], [7.80, 9.45]] as [number, number][],
      ],
    },
  ];

  return (
    <>
      {confluences.map((c, idx) => (
        <div key={idx}>
          {/* Dual layers of Polylines for glowing water road effect */}
          {c.rivers.map((r, rIdx) => (
            <div key={`river-${rIdx}`}>
              {/* Thick cyan glowing base */}
              <Polyline
                positions={r}
                pane="confluencePane"
                pathOptions={{
                  color: "#06b6d4",
                  weight: 8,
                  opacity: 0.35,
                  className: "confluence-glow-base",
                }}
              />
              {/* Thin, vibrant animated core */}
              <Polyline
                positions={r}
                pane="confluencePane"
                pathOptions={{
                  color: "#22d3ee",
                  weight: 3,
                  opacity: 0.9,
                  dashArray: "10,12",
                  className: "confluence-flow-line",
                }}
              />
            </div>
          ))}

          {/* Radial outer glow zone circle */}
          <Circle
            center={c.center}
            radius={3500}
            pane="confluencePane"
            pathOptions={{
              fillColor: "#0891b2",
              fillOpacity: 0.18,
              color: "transparent",
              weight: 0,
            }}
          />

          {/* Glowing junction center with scientific wave displacement filter */}
          <Circle
            center={c.center}
            radius={1800}
            pane="confluencePane"
            pathOptions={{
              fillColor: "#22d3ee",
              fillOpacity: 0.85,
              color: "#ffffff",
              weight: 1.5,
              dashArray: "3,3",
              className: "confluence-pulsing-core",
            }}
          >
            <Tooltip permanent direction="top" offset={[0, -6]} className="!bg-[#0f172a]/95 !border !border-cyan-500/40 !shadow-2xl !text-white !font-bold !text-[9px] !rounded-lg !px-2.5 !py-1.5">
              <div className="flex flex-col gap-0.5">
                <span className="text-cyan-400 font-extrabold flex items-center gap-1">🌊 {c.name}</span>
                <span className="text-white/70 block text-[8px] font-semibold">{c.lga} • {c.state} State</span>
                <span className="text-[7.5px] text-cyan-200/50 uppercase tracking-widest mt-0.5 font-bold">Scientific Confluence Zone</span>
              </div>
            </Tooltip>
          </Circle>
        </div>
      ))}
    </>
  );
};

const IrrigationLegend = () => (
  <div
    className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 540 }}
  >
    <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">
      Agro-Ecological Irrigation Zones & Water Confluences
    </div>
    
    <div className="flex gap-4 mb-3">
      {/* Arid Zones */}
      <div className="flex-1 flex flex-col gap-1">
        <div className="text-[9px] font-bold text-center text-orange-200">ARID ZONES</div>
        <div className="h-2 w-full rounded-full bg-gradient-to-r from-yellow-300 to-orange-500" />
        <div className="flex justify-between text-[8px] text-white/60">
          <span>Single Crop</span><span>Multi-crop</span>
        </div>
      </div>
      
      {/* Dry Season Zones */}
      <div className="flex-1 flex flex-col gap-1">
        <div className="text-[9px] font-bold text-center text-lime-200">DRY SEASON ZONES</div>
        <div className="h-2 w-full rounded-full bg-gradient-to-r from-lime-400 to-[#4d7c0f]" />
        <div className="flex justify-between text-[8px] text-white/60">
          <span>Single Crop</span><span>Multi-crop</span>
        </div>
      </div>

      {/* No Dry Season Zones */}
      <div className="flex-1 flex flex-col gap-1">
        <div className="text-[9px] font-bold text-center text-cyan-200">NO DRY SEASON</div>
        <div className="h-2 w-full rounded-full bg-gradient-to-r from-cyan-400 to-blue-700" />
        <div className="flex justify-between text-[8px] text-white/60">
          <span>Single Crop</span><span>Multi-crop</span>
        </div>
      </div>
    </div>

    {/* Confluences explanation */}
    <div className="border-t border-white/10 pt-2 flex items-center justify-between text-[9px] text-white/60">
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full bg-cyan-500 border border-white/50" />
        <span className="font-semibold text-white/80">Major River Confluence</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="w-6 h-1 bg-cyan-400 rounded" style={{ borderTop: "2px dashed #22d3ee" }} />
        <span>Animated Flowing Channels (Water Roads)</span>
      </div>
      <span className="text-[8px] text-cyan-400/80">Supply Confluences attached to nearest LGA</span>
    </div>
  </div>
);
