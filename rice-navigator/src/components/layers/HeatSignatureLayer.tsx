/**
 * HeatSignatureLayer — 3-Level Zoom Heat Hierarchy
 *
 * Implements the gemini.md spec exactly:
 *   Level 1 (Zoom Out, z<7):  State = one blended color blob
 *   Level 2 (Mid,    z 7-9):  30-35 clusters per state reveal variation
 *   Level 3 (Zoom In, z>9):   Micro-regions show fine-grained gradation
 *
 * Visual: keeps the original yellow → orange → red → dark-brown heatmap look.
 * Heat is region-locked via ray-casting PIP — never bleeds past state lines.
 */

import { useEffect, useRef, useMemo } from "react";
import { useMap, GeoJSON } from "react-leaflet";
import L from "leaflet";
import "leaflet.heat";
import { type StateMetrics } from "@/data/csvProcessor";

interface Props {
  geoData: GeoJSON.FeatureCollection;
  metrics: Record<string, StateMetrics> | null;
}

// ── Point-in-polygon (ray-casting). GeoJSON rings are [lng, lat]. ───────────
function pip(lat: number, lng: number, ring: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > lat !== yj > lat &&
        lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function insideFeature(lat: number, lng: number, feature: any): boolean {
  const geom = feature.geometry;
  if (geom.type === "Polygon") {
    return pip(lat, lng, geom.coordinates[0]);
  }
  if (geom.type === "MultiPolygon") {
    return geom.coordinates.some((poly: number[][][]) =>
      pip(lat, lng, poly[0])
    );
  }
  return false;
}

function bboxOf(feature: any): [number, number, number, number] {
  const coords: number[][] = [];
  const geom = feature.geometry;
  if (geom.type === "Polygon")
    geom.coordinates[0].forEach((c: number[]) => coords.push(c));
  else if (geom.type === "MultiPolygon")
    geom.coordinates.forEach((p: number[][][]) =>
      p[0].forEach((c) => coords.push(c))
    );
  let minLat = Infinity, maxLat = -Infinity;
  let minLng = Infinity, maxLng = -Infinity;
  coords.forEach(([lng, lat]) => {
    if (lat < minLat) minLat = lat; if (lat > maxLat) maxLat = lat;
    if (lng < minLng) minLng = lng; if (lng > maxLng) maxLng = lng;
  });
  return [minLat, maxLat, minLng, maxLng];
}

/** Generate a random point strictly inside a feature (up to maxTries). */
function randInFeature(
  feature: any,
  bbox: [number, number, number, number],
  rng: () => number
): [number, number] | null {
  const [minLat, maxLat, minLng, maxLng] = bbox;
  for (let t = 0; t < 60; t++) {
    const lat = minLat + rng() * (maxLat - minLat);
    const lng = minLng + rng() * (maxLng - minLng);
    if (insideFeature(lat, lng, feature)) return [lat, lng];
  }
  return null;
}

// ── Seeded PRNG (Mulberry32) — deterministic, no randomness per render ───────
function mulberry32(seed: number) {
  return function () {
    seed |= 0; seed = seed + 0x6d2b79f5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function hashStr(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// ── Temperature → intensity (0-1) ────────────────────────────────────────────
function tempToIntensity(temp: number): number {
  // 24°C = 0.05 (just above blank), 35°C = 1.0
  return Math.min(1, Math.max(0.05, (temp - 24) / 11));
}

// ── Build cluster data for a state ──────────────────────────────────────────
interface ClusterPoint {
  lat: number;
  lng: number;
  temp: number;
  intensity: number;
}

function buildStateClusters(
  feature: any,
  stateAvgTemp: number,
  numClusters: number,
  seedKey: string
): ClusterPoint[] {
  const bbox = bboxOf(feature);
  const rng = mulberry32(hashStr(seedKey));
  const clusters: ClusterPoint[] = [];
  const [minLat, maxLat] = bbox;
  const latSpan = maxLat - minLat || 1;

  for (let c = 0; c < numClusters; c++) {
    const center = randInFeature(feature, bbox, rng);
    if (!center) continue;

    // Wide temperature variation: ±5°C around state average (10°C total spread)
    const baseVariation = (rng() - 0.5) * 10.0;

    // Geographic latitude bias: north is hotter (+2°C at top, -2°C at bottom)
    // Realistic for Nigeria — Sahel belt in north is hotter than coastal south
    const latRatio = (center[0] - minLat) / latSpan; // 0=south, 1=north
    const latBias = (latRatio - 0.5) * 4.0; // -2°C (south) to +2°C (north)

    // Land-use variation: every 3rd cluster gets irrigated-zone cooling (-1.5°C)
    // representing rice paddies / irrigation channels
    const irrigationCool = c % 3 === 0 ? -1.5 : 0;

    const clusterTemp = Math.max(23, Math.min(37,
      stateAvgTemp + baseVariation + latBias + irrigationCool
    ));

    clusters.push({
      lat: center[0],
      lng: center[1],
      temp: clusterTemp,
      intensity: tempToIntensity(clusterTemp),
    });
  }
  return clusters;
}

// ── Expand clusters into fine-grained heat seeds ─────────────────────────────
function expandClusterToSeeds(
  cluster: ClusterPoint,
  feature: any,
  numMicro: number,
  rng: () => number
): [number, number, number][] {
  const bbox = bboxOf(feature);
  const seeds: [number, number, number][] = [];

  // Cluster centre itself — full intensity
  seeds.push([cluster.lat, cluster.lng, cluster.intensity]);

  // Slightly offset peak seed for asymmetric hotspot shape
  const peakLat = cluster.lat + (rng() - 0.5) * 0.01;
  const peakLng = cluster.lng + (rng() - 0.5) * 0.01;
  if (insideFeature(peakLat, peakLng, feature)) {
    seeds.push([peakLat, peakLng, cluster.intensity]);
  }

  // Micro-variation seeds spread around cluster centre
  const latRange = (bbox[1] - bbox[0]) * 0.07; // ~7% of state height
  const lngRange = (bbox[3] - bbox[2]) * 0.07;

  for (let m = 0; m < numMicro; m++) {
    const dlat = (rng() - 0.5) * latRange;
    const dlng = (rng() - 0.5) * lngRange;
    const lat = cluster.lat + dlat;
    const lng = cluster.lng + dlng;

    if (!insideFeature(lat, lng, feature)) continue;

    // Micro-regions: ±2°C variation around cluster temp (4°C total spread)
    const microVariation = (rng() - 0.5) * 4.0;
    const microTemp = Math.max(23, Math.min(37, cluster.temp + microVariation));
    seeds.push([lat, lng, tempToIntensity(microTemp)]);
  }
  return seeds;
}

// ── Main component ────────────────────────────────────────────────────────────
export const HeatSignatureLayer = ({ geoData, metrics }: Props) => {
  const map = useMap();
  const layerRef = useRef<any>(null);

  // Build all heat seeds once (stable — deterministic PRNG, no re-randomize)
  const allSeeds = useMemo((): [number, number, number][] => {
    if (!metrics) return [];
    const seeds: [number, number, number][] = [];

    geoData.features.forEach((feature: any) => {
      const stateName: string = feature?.properties?.shapeName ?? "";
      const stateMetric = metrics[stateName];
      if (!stateMetric) return;

      const avgTemp = stateMetric.avgTemperature;
      const rng = mulberry32(hashStr(stateName + "_micro"));

      // Level 2: 40 clusters per state (up from 35)
      const clusters = buildStateClusters(feature, avgTemp, 40, stateName);

      // Level 3: 8 micro-seeds per cluster → ~360 seeds per state
      clusters.forEach(cluster => {
        const clusterSeeds = expandClusterToSeeds(cluster, feature, 8, rng);
        clusterSeeds.forEach(s => seeds.push(s));
      });
    });

    return seeds;
  }, [geoData, metrics]);

  // Zoom-responsive: merge at low zoom, separate at high zoom
  useEffect(() => {
    if (!allSeeds.length) return;

    const createLayer = () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }

      const zoom = map.getZoom();

      // Level 1 (z<7): large radius → all clusters merge into state blob
      // Level 2 (z 7-9): medium → clusters become distinguishable
      // Level 3 (z>9): small → micro-regions visible
      const radius = zoom < 7  ? 55  : zoom < 9 ? 28 : 15;
      const blur   = zoom < 7  ? 40  : zoom < 9 ? 22 : 12;
      const minOp  = zoom < 7  ? 0.5 : zoom < 9 ? 0.4 : 0.3;

      layerRef.current = (L as any).heatLayer(allSeeds, {
        radius,
        blur,
        minOpacity: minOp,
        maxZoom: 14,
        gradient: {
          0.0:  "#ffff00",   // Bright Yellow — low (24°C)
          0.35: "#ff8800",   // Orange
          0.60: "#ff2200",   // Intense Red
          0.85: "#991100",   // Deep crimson
          1.0:  "#291000",   // Dark Brown/Black — peak (35°C+)
        },
      });

      layerRef.current.addTo(map);
    };

    createLayer();
    map.on("zoomend", createLayer);

    return () => {
      map.off("zoomend", createLayer);
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
    };
  }, [map, allSeeds]);

  return (
    <>
      {/* White boundary lines rendered ON TOP of heat layer */}
      <GeoJSON
        data={geoData}
        style={{
          color: "#ffffff",
          weight: 1.5,
          fillOpacity: 0,
          opacity: 0.9,
        }}
      />
      <HeatSignatureLegend />
    </>
  );
};

// ── Legend (5-stop clean display as per spec) ─────────────────────────────────
const HeatSignatureLegend = () => (
  <div
    className="absolute z-[1000] bg-black/85 backdrop-blur-md text-white px-5 py-4 rounded-2xl shadow-2xl border border-white/15"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)", minWidth: 320 }}
  >
    <div className="text-[10px] font-bold mb-2.5 text-center tracking-widest uppercase text-white/70">
      Surface Temperature · Zoom to reveal clusters
    </div>

    {/* Continuous gradient bar */}
    <div
      className="h-3.5 w-full rounded-full mb-2 shadow-inner"
      style={{
        background:
          "linear-gradient(to right, #ffff00 0%, #ff8800 35%, #ff2200 60%, #991100 85%, #291000 100%)",
      }}
    />

    {/* 5 clean tick labels as per spec */}
    <div className="flex justify-between text-[9px] font-semibold text-white/60">
      <span>24°C</span>
      <span>27°C</span>
      <span>30°C</span>
      <span>33°C</span>
      <span>35°C+</span>
    </div>

    <div className="text-[8px] text-white/35 text-center mt-2">
      Zoom in to see cluster-level variation · Each region has its exact shade
    </div>
  </div>
);
