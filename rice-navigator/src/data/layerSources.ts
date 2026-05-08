/**
 * GIS Layer Sources — circular blobs centered on real rice field positions.
 * Values derived from the real CSV dataset (csvStateData).
 */
import { riceFields, csvStateData } from "./nigeriaRiceData";

export type LayerKey =
  | "healthy"
  | "ndvi"
  | "rainfall"
  | "temperature"
  | "risk"
  | "irrigation"
  | "yield";

export interface LayerMeta {
  key: LayerKey;
  label: string;
  shortLabel: string;
  color: string;
  description: string;
}

export const LAYERS: LayerMeta[] = [
  { key: "healthy",     label: "Healthy Fields",      shortLabel: "Healthy",    color: "#22c55e", description: "Active vegetation cover (NDVI > 0.45)" },
  { key: "ndvi",        label: "NDVI / EVI Vigor",    shortLabel: "NDVI",       color: "#eab308", description: "Vegetation vigor index" },
  { key: "rainfall",    label: "Rainfall Patterns",   shortLabel: "Rainfall",   color: "#3b82f6", description: "Daily average rainfall (mm)" },
  { key: "temperature", label: "Temperature Stress",  shortLabel: "Temp",       color: "#f97316", description: "Heat stress zones (> 29°C)" },
  { key: "risk",        label: "Risk Zones",          shortLabel: "Risk",       color: "#dc2626", description: "Drought / flood risk index" },
  { key: "irrigation",  label: "Irrigation Network",  shortLabel: "Irrigation", color: "#0ea5e9", description: "Water availability (NDWI)" },
  { key: "yield",       label: "Yield Forecast",      shortLabel: "Yield",      color: "#8b5cf6", description: "Predicted t/ha from NDVI" },
];

// ─── Circle polygon generator ────────────────────────────────────────────────
// Approximates a circle as an N-point GeoJSON polygon.
// radiusDeg: radius in degrees (≈ 1° lat = 111 km)
function makeCircle(
  lat: number,
  lng: number,
  radiusDeg: number,
  value: number,
  points = 64
): GeoJSON.Feature {
  const coords = Array.from({ length: points + 1 }, (_, i) => {
    const angle = (i / points) * 2 * Math.PI;
    // Adjust lng radius for latitude distortion
    const lngR = radiusDeg / Math.cos((lat * Math.PI) / 180);
    return [lng + lngR * Math.cos(angle), lat + radiusDeg * Math.sin(angle)];
  });
  return {
    type: "Feature",
    properties: { value },
    geometry: { type: "Polygon", coordinates: [coords] },
  };
}

// ─── Value normalizers (all → 0..1) ─────────────────────────────────────────
const normNdvi  = (v: number) => Math.min(1, v / 0.6);
const normRain  = (v: number) => Math.min(1, v / 10);
const normTemp  = (v: number) => Math.min(1, Math.max(0, (v - 24) / 10));
const normNdwi  = (v: number) => Math.min(1, Math.max(0, (v + 0.6) / 0.6));

// Radius scales slightly with hectares so larger farms get bigger circles
const radiusForField = (hectares: number) =>
  0.55 + Math.sqrt(hectares) / 120;

// ─── Build a FeatureCollection of circles for a given field subset + value fn ─
function circlesFor(
  fieldTypes: Array<"healthy" | "stress" | "irrigation" | "growth">,
  valueFn: (state: string) => number
): GeoJSON.FeatureCollection {
  const features = riceFields
    .filter((f) => fieldTypes.includes(f.type) && f.state in csvStateData)
    .map((f) =>
      makeCircle(
        f.position[0],
        f.position[1],
        radiusForField(f.hectares),
        valueFn(f.state)
      )
    );
  return { type: "FeatureCollection", features };
}

// ─── Layer sources ────────────────────────────────────────────────────────────
export const layerSources: Record<LayerKey, () => GeoJSON.FeatureCollection> = {
  healthy: () =>
    circlesFor(["healthy"], (s) => normNdvi(csvStateData[s].ndvi)),

  ndvi: () =>
    circlesFor(["healthy", "growth", "irrigation", "stress"], (s) =>
      normNdvi(csvStateData[s].ndvi)
    ),

  rainfall: () =>
    circlesFor(["healthy", "growth", "irrigation", "stress"], (s) =>
      normRain(csvStateData[s].rainfall)
    ),

  temperature: () =>
    circlesFor(["stress", "healthy", "growth", "irrigation"], (s) =>
      normTemp(csvStateData[s].temperature)
    ),

  risk: () =>
    circlesFor(["stress"], (s) => 1 - normNdvi(csvStateData[s].ndvi)),

  irrigation: () =>
    circlesFor(["irrigation", "healthy", "growth"], (s) =>
      normNdwi(csvStateData[s].ndwi)
    ),

  yield: () =>
    circlesFor(["healthy", "growth"], (s) =>
      normNdvi(csvStateData[s].ndvi)
    ),
};

// Color helper (kept for compatibility)
export function rampColor(hex: string, value: number): string {
  const op = 0.15 + value * 0.55;
  return hex + Math.round(op * 255).toString(16).padStart(2, "0");
}
