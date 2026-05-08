import { useMemo, useRef } from "react";
import { CircleMarker, GeoJSON } from "react-leaflet";
import L from "leaflet";
import { type StateMetrics, type DailyRecord } from "@/data/csvProcessor";

interface Props {
  geoData: GeoJSON.FeatureCollection;
  metrics: Record<string, StateMetrics> | null;
  rawRecords: DailyRecord[];
}

/**
 * Point-in-polygon test (ray-casting) for [lng, lat] rings.
 */
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
  // Fallback: polygon centroid approximation
  return [(minLat + maxLat) / 2, (minLng + maxLng) / 2];
}

/** Build a map of stateName → GeoJSON feature for fast lookup */
function buildFeatureIndex(geoData: GeoJSON.FeatureCollection) {
  const idx = new Map<string, any>();
  geoData.features.forEach(f => {
    const name = (f as any).properties?.shapeName;
    if (name) idx.set(name, f);
  });
  return idx;
}

/** Chronological colour ramp: Magenta → Blue → Teal → Green */
function getColor(ratio: number): string {
  if (ratio < 0.25) return "#d946ef"; // Magenta (Earliest)
  if (ratio < 0.50) return "#1e3a8a"; // Dark Blue
  if (ratio < 0.75) return "#0f766e"; // Teal
  return "#4ade80";                   // Light Green (Latest)
}

export const TemporalScatterLayer = ({ geoData, rawRecords }: Props) => {
  const pointCacheRef = useRef<Map<string, [number, number][]>>(new Map());

  const scatterPoints = useMemo(() => {
    if (!rawRecords.length) return [];

    const featureIndex = buildFeatureIndex(geoData);

    // Find overall date range for ratio calc
    const allDates = rawRecords.map(r => r.date).sort();
    const startTime = new Date(allDates[0]).getTime();
    const endTime   = new Date(allDates[allDates.length - 1]).getTime();
    const span      = endTime - startTime || 1;

    // Filter to records where actual rain fell (daily mm > 0.3)
    const rainyRecords = rawRecords.filter(r => r.precipitation > 0.3);

    // Build scatter — each record → one placed point within its state polygon
    const points: { lat: number; lng: number; dateRatio: number; radius: number }[] = [];

    // Group records by state to limit per-state volume
    const byState: Record<string, DailyRecord[]> = {};
    for (const r of rainyRecords) {
      (byState[r.state] ??= []).push(r);
    }

    for (const [stateName, stateRecords] of Object.entries(byState)) {
      const feature = featureIndex.get(stateName);
      if (!feature) continue;

      // Initialise point cache for this state
      if (!pointCacheRef.current.has(stateName)) {
        // Pre-generate 300 polygon-clipped positions for this state
        const positions: [number, number][] = [];
        for (let k = 0; k < 300; k++) {
          const pt = randomPointInFeature(feature);
          if (pt) positions.push(pt);
        }
        pointCacheRef.current.set(stateName, positions);
      }
      const positions = pointCacheRef.current.get(stateName)!;

      // Sample up to 200 events per state to keep perf sane
      const sample = stateRecords.length > 200
        ? stateRecords.filter((_, i) => i % Math.ceil(stateRecords.length / 200) === 0)
        : stateRecords;

      sample.forEach((r, i) => {
        const [lat, lng] = positions[i % positions.length];
        const dateRatio = (new Date(r.date).getTime() - startTime) / span;
        // Scale radius to rain intensity (0.3–20mm range → 2–5px)
        const radius = 2 + Math.min(3, (r.precipitation / 10) * 3);
        points.push({ lat, lng, dateRatio, radius });
      });
    }

    return points;
  }, [geoData, rawRecords]);

  return (
    <>
      {/* Thick black state outlines over the scatter */}
      <GeoJSON
        data={geoData}
        style={{ color: "#111111", weight: 2.5, fillOpacity: 0, opacity: 0.9 }}
      />

      {/* Scatter points */}
      {scatterPoints.map((pt, i) => (
        <CircleMarker
          key={i}
          center={[pt.lat, pt.lng]}
          radius={pt.radius}
          pathOptions={{
            color: "transparent",
            fillColor: getColor(pt.dateRatio),
            fillOpacity: 0.55,
            weight: 0,
          }}
        />
      ))}

      <TemporalLegend count={scatterPoints.length} />
    </>
  );
};

const TemporalLegend = ({ count }: { count: number }) => (
  <div
    className="absolute z-[1000] bg-white/95 backdrop-blur-sm px-4 py-3 rounded-xl shadow-2xl border border-gray-200"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)" }}
  >
    <div className="text-[11px] font-bold mb-1 text-gray-700 text-center tracking-widest uppercase">
      Temporal Rainfall Distribution
    </div>
    <div className="text-[9px] text-center text-gray-400 mb-2">
      {count.toLocaleString()} precipitation events · each dot = 1 day
    </div>
    <div className="flex items-center gap-4 justify-center">
      {[
        { color: "#d946ef", label: "Earliest" },
        { color: "#1e3a8a", label: "Mid" },
        { color: "#0f766e", label: "Late" },
        { color: "#4ade80", label: "Latest" },
      ].map(({ color, label }) => (
        <div key={label} className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full border border-gray-300" style={{ backgroundColor: color }} />
          <span className="text-[9px] font-semibold text-gray-600">{label}</span>
        </div>
      ))}
    </div>
  </div>
);
