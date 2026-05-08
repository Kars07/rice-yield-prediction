import { useEffect, useMemo, useRef } from "react";
import { useMap, GeoJSON } from "react-leaflet";
import L from "leaflet";
import "leaflet.heat";
import { type StateMetrics } from "@/data/csvProcessor";

interface Props {
  geoData: GeoJSON.FeatureCollection;
  metrics: Record<string, StateMetrics> | null;
}

/**
 * Generates random points strictly INSIDE a GeoJSON polygon using
 * a point-in-polygon test — so the heat only appears within Nigeria.
 */
function randomPointInPolygon(feature: any, count: number): [number, number][] {
  const coords: [number, number][] = [];

  // Flatten all rings from Polygon or MultiPolygon
  const rings: number[][][] = [];
  const geom = feature.geometry;
  if (geom.type === "Polygon") {
    rings.push(geom.coordinates[0]);
  } else if (geom.type === "MultiPolygon") {
    geom.coordinates.forEach((poly: number[][][]) => rings.push(poly[0]));
  }
  if (!rings.length) return coords;

  // Compute bounding box
  let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
  rings.forEach(ring =>
    ring.forEach(([lng, lat]) => {
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
    })
  );

  // Simple ray-casting point-in-polygon for the outer ring
  const pip = (lat: number, lng: number, ring: number[][]): boolean => {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      const intersect =
        yi > lat !== yj > lat &&
        lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  };

  let attempts = 0;
  while (coords.length < count && attempts < count * 20) {
    attempts++;
    const lat = minLat + Math.random() * (maxLat - minLat);
    const lng = minLng + Math.random() * (maxLng - minLng);
    // Check if point is inside any ring
    if (rings.some(ring => pip(lat, lng, ring))) {
      coords.push([lat, lng]);
    }
  }
  return coords;
}

export const HeatSignatureLayer = ({ geoData, metrics }: Props) => {
  const map = useMap();
  // Stable seed for points so they don't re-randomize on every re-render
  const pointsRef = useRef<Map<string, [number, number][]>>(new Map());

  // Pre-generate stable polygon-clipped points once per feature
  const heatData = useMemo(() => {
    if (!metrics) return [];
    const points: [number, number, number][] = [];

    geoData.features.forEach((feature: any) => {
      const stateName = feature.properties.shapeName;
      const stateMetric = metrics[stateName];
      if (!stateMetric) return;

      const temp = stateMetric.avgTemperature;
      const intensity = Math.min(1, Math.max(0.15, (temp - 24) / 10));

      // Cache points so they're stable across re-renders
      if (!pointsRef.current.has(stateName)) {
        pointsRef.current.set(stateName, randomPointInPolygon(feature, 60));
      }
      const statePoints = pointsRef.current.get(stateName)!;
      statePoints.forEach(([lat, lng]) => {
        points.push([lat, lng, intensity]);
      });
    });
    return points;
  }, [geoData, metrics]);

  useEffect(() => {
    if (!heatData.length) return;

    const heatLayer = (L as any).heatLayer(heatData, {
      radius: 30,       // Smaller radius → tighter, state-scoped blobs
      blur: 25,
      minOpacity: 0.4,
      maxZoom: 8,
      gradient: {
        0.0: "#ffff00",  // Bright Yellow (Low)
        0.5: "#ff4400",  // Orange-Red (Medium)
        0.8: "#ff0000",  // Intense Red (High)
        1.0: "#291000",  // Dark Brown/Black (Peak)
      },
    }).addTo(map);

    return () => {
      map.removeLayer(heatLayer);
    };
  }, [map, heatData]);

  return (
    <>
      {/* Thin black boundary lines for geographic context — rendered ON TOP of heat */}
      <GeoJSON
        data={geoData}
        style={{
          color: "#1a1a1a",
          weight: 1.5,
          fillOpacity: 0,
          opacity: 1,
        }}
      />
      <HeatSignatureLegend />
    </>
  );
};

const HeatSignatureLegend = () => (
  <div
    className="absolute z-[1000] bg-black/80 text-white px-4 py-3 rounded-xl shadow-2xl border border-white/20"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 280 }}
  >
    <div className="text-[11px] font-bold mb-2 text-center tracking-widest uppercase opacity-80">
      Heat Signatures — Surface Temperature
    </div>
    {/* Horizontal gradient bar */}
    <div
      className="h-3 w-full rounded-full mb-1"
      style={{
        background: "linear-gradient(to right, #ffff00, #ff4400, #ff0000, #291000)",
      }}
    />
    {/* Tick labels */}
    <div className="flex justify-between text-[9px] text-gray-300 font-semibold">
      <span>24°C</span>
      <span>27°C</span>
      <span>30°C</span>
      <span>34°C+</span>
    </div>
  </div>
);
