/**
 * MapDrawControl — wraps react-leaflet-draw's EditControl.
 * When the user finishes drawing a polygon the component derives mock zonal
 * statistics from whichever state the polygon centroid is closest to, then
 * shows them in a Leaflet popup bound to the drawn layer.
 */
import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "@geoman-io/leaflet-geoman-free";
import "@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css";
import { csvStateData } from "@/data/nigeriaRiceData";
import type { StateMetrics } from "@/data/csvProcessor";

// Small ± variation so repeated clicks feel different
const jitter = (base: number, range: number) =>
  parseFloat((base + (Math.random() * 2 - 1) * range).toFixed(2));

function closestState(lat: number, lng: number): string {
  // Rough centroids for each state that has CSV data
  const centroids: Record<string, [number, number]> = {
    Kebbi:  [12.45, 4.20],
    Kano:   [11.70, 8.52],
    Niger:  [9.60,  6.55],
    Ebonyi: [6.32,  8.09],
    Taraba: [8.00, 10.77],
    Jigawa: [12.21, 9.35],
  };
  let best = "Niger";
  let bestDist = Infinity;
  for (const [state, [cLat, cLng]] of Object.entries(centroids)) {
    const d = Math.hypot(lat - cLat, lng - cLng);
    if (d < bestDist) { bestDist = d; best = state; }
  }
  return best;
}

interface Props {
  csvMetrics: Record<string, StateMetrics> | null;
}

export default function MapDrawControl({ csvMetrics }: Props) {
  const map = useMap();
  const drawnLayersRef = useRef<L.FeatureGroup | null>(null);

  useEffect(() => {
    // Create a feature group for drawn shapes
    const drawnLayer = new L.FeatureGroup();
    drawnLayer.addTo(map);
    drawnLayersRef.current = drawnLayer;

    // Initialize Geoman controls
    map.pm.addControls({
      position: 'topleft',
      drawCircle: false,
      drawMarker: false,
      drawCircleMarker: false,
      drawPolyline: false,
      drawText: false,
      drawPolygon: true,
      drawRectangle: true,
      editMode: true,
      dragMode: true,
      cutPolygon: false,
      removalMode: true,
    });
    
    // Configure default drawing styles
    map.pm.setPathOptions({
      color: "#8b5cf6",
      weight: 2,
      fillOpacity: 0.15,
    });

    // Handler: polygon created
    const onCreate = (e: any) => {
      const layer: L.Layer = e.layer;
      drawnLayer.addLayer(layer);

      // Find centroid (rough average of coords)
      let lat = 0, lng = 0, count = 0;
      if ((layer as any).getLatLngs) {
        const latlngs: L.LatLng[] = (layer as any).getLatLngs()[0] as L.LatLng[];
        for (const ll of latlngs) { lat += ll.lat; lng += ll.lng; count++; }
        lat /= count; lng /= count;
      } else if ((layer as any).getLatLng) {
        const ll = (layer as any).getLatLng() as L.LatLng;
        lat = ll.lat; lng = ll.lng;
      }

      const state = closestState(lat, lng);
      const data = csvStateData[state];
      const met  = csvMetrics?.[state];

      const ndvi  = data ? jitter(data.ndvi, 0.03) : 0.40;
      const temp  = data ? jitter(data.temperature, 1.2) : 28.0;
      const rain  = data ? jitter(data.rainfall, 0.5) : 5.0;
      const yield_ = parseFloat((ndvi * 6.5).toFixed(2));
      const healthColor = ndvi >= 0.45 ? "#22c55e" : ndvi >= 0.30 ? "#eab308" : "#ef4444";
      const healthLabel = ndvi >= 0.45 ? "Healthy" : ndvi >= 0.30 ? "Moderate" : "Stressed";
      const phenology = met?.phenologyStage ?? "Vegetative";

      const popupHtml = `
        <div style="font-family:Inter,sans-serif;min-width:220px;padding:4px 2px">
          <div style="font-weight:700;font-size:13px;margin-bottom:8px;color:#1e293b">
            📐 Zone Analysis — ${state} State
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px">
            <div style="background:#f1f5f9;border-radius:8px;padding:8px">
              <div style="font-size:9px;color:#64748b;font-weight:600;text-transform:uppercase;margin-bottom:2px">Est. NDVI</div>
              <div style="font-size:15px;font-weight:700;color:#1e293b">${ndvi}</div>
            </div>
            <div style="background:#f1f5f9;border-radius:8px;padding:8px">
              <div style="font-size:9px;color:#64748b;font-weight:600;text-transform:uppercase;margin-bottom:2px">Yield Est.</div>
              <div style="font-size:15px;font-weight:700;color:#1e293b">${yield_} t/ha</div>
            </div>
            <div style="background:#f1f5f9;border-radius:8px;padding:8px">
              <div style="font-size:9px;color:#64748b;font-weight:600;text-transform:uppercase;margin-bottom:2px">Avg Temp</div>
              <div style="font-size:15px;font-weight:700;color:#1e293b">${temp.toFixed(1)}°C</div>
            </div>
            <div style="background:#f1f5f9;border-radius:8px;padding:8px">
              <div style="font-size:9px;color:#64748b;font-weight:600;text-transform:uppercase;margin-bottom:2px">Rainfall</div>
              <div style="font-size:15px;font-weight:700;color:#1e293b">${rain.toFixed(1)} mm/d</div>
            </div>
          </div>
          <div style="display:flex;gap:6px">
            <span style="background:${healthColor}22;color:${healthColor};font-size:10px;font-weight:700;padding:3px 8px;border-radius:999px">
              ${healthLabel}
            </span>
            <span style="background:#8b5cf622;color:#8b5cf6;font-size:10px;font-weight:700;padding:3px 8px;border-radius:999px">
              ${phenology}
            </span>
          </div>
        </div>
      `;

      (layer as L.Layer & { bindPopup: (h: string, o?: any) => any }).bindPopup(popupHtml, {
        maxWidth: 280,
        className: "draw-zone-popup",
      }).openPopup();
    };

    map.on('pm:create', onCreate);

    return () => {
      map.off('pm:create', onCreate);
      map.pm.removeControls();
      map.removeLayer(drawnLayer);
    };
  }, [map, csvMetrics]);

  return null;
}
