import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, GeoJSON, useMap, Tooltip, Polyline, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { NIGERIA_CENTER, NIGERIA_BOUNDS, csvStateData, mockIrrigationJunctions, type RiceField } from "@/data/nigeriaRiceData";
import { LAYERS, layerSources, type LayerKey } from "@/data/layerSources";
import type { MapStyle } from "@/components/MapLayersControl";
import MapLegend from "@/components/MapLegend";
import MapDrawControl from "@/components/MapDrawControl";
import AdvancedRasterLayer from "@/components/AdvancedRasterLayer";
import { HeatSignatureLayer } from "@/components/layers/HeatSignatureLayer";
import { TemporalScatterLayer } from "@/components/layers/TemporalScatterLayer";
import { WaterStressChoropleth } from "@/components/layers/WaterStressChoropleth";
import { statePassesFilter, type FeatureKey, type StateMetrics } from "@/data/csvProcessor";

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

const typeColors: Record<RiceField["type"], string> = {
  healthy: "#22c55e",
  stress: "#ef4444",
  irrigation: "#0ea5e9",
  growth: "#eab308",
};

const layerToFieldTypes: Record<LayerKey, RiceField["type"][]> = {
  healthy:     ["healthy"],
  ndvi:        ["healthy", "growth"],
  rainfall:    ["healthy", "growth", "irrigation"],
  temperature: ["stress"],
  risk:        ["stress"],
  irrigation:  ["irrigation"],
  yield:       ["healthy", "growth"],
};

// Circle marker with a map-pin icon inside, color-matched to the field type
// Circle marker with an emoji inside, color-matched to the field type
const markerSvg = (color: string, selected: boolean, type: RiceField["type"]) => {
  const emojiMap: Record<RiceField["type"], string> = { healthy: "🌱", stress: "🍂", irrigation: "💧", growth: "🌾" };
  const emoji = emojiMap[type] || "📍";
  const size = selected ? 44 : 34;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 3;

  // Pulse ring for selected state
  const ring = selected
    ? `<circle cx="${cx}" cy="${cy}" r="${size / 2 - 1}" fill="none" stroke="${color}" stroke-opacity="0.30" stroke-width="7"/>`
    : "";

  return (
    `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">` +
    ring +
    `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${color}" stroke="white" stroke-width="2.5"/>` +
    `<text x="${cx}" y="${cy + 2}" font-size="${r * 1.2}" font-family="Arial" text-anchor="middle" dominant-baseline="middle">${emoji}</text>` +
    `</svg>`
  );
};

// ─── Irrigation Junction icon ─────────────────────────────────────────────────
const junctionSvg = (status: "Good" | "Low" | "Critical") => {
  const color = status === "Good" ? "#0ea5e9" : status === "Low" ? "#f97316" : "#ef4444";
  return (
    `<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">` +
    `<circle cx="16" cy="16" r="13" fill="${color}" stroke="white" stroke-width="2.5"/>` +
    `<svg x="8" y="8" width="16" height="16" viewBox="0 0 24 24" fill="none">` +
    `<path d="M12 2 Q7 9 7 13a5 5 0 0010 0Q17 9 12 2z" fill="white"/>` +
    `</svg>` +
    `</svg>`
  );
};

const junctionIconCache = new Map<string, L.DivIcon>();
const getJunctionIcon = (status: "Good" | "Low" | "Critical"): L.DivIcon => {
  if (junctionIconCache.has(status)) return junctionIconCache.get(status)!;
  const icon = L.divIcon({
    html: junctionSvg(status),
    className: "",
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
  junctionIconCache.set(status, icon);
  return icon;
};

interface NigeriaMapProps {
  activeLayer: LayerKey;
  activeFeature: FeatureKey;
  csvMetrics: Record<string, StateMetrics> | null;
  mapStyle: MapStyle;
  mapView: "state" | "farm";
  selectedField: RiceField | null;
  selectedState: string | null;
  flyTo: [number, number] | null;
  flyToZoom?: number;
  riceFields: RiceField[];
  rawRecords: import("@/data/csvProcessor").DailyRecord[];
  onFieldClick: (field: RiceField) => void;
  onStateClick: (stateName: string) => void;
  onMapClick: () => void;
}

const iconCache = new Map<string, L.DivIcon>();
const getMarkerIcon = (field: RiceField, isSelected: boolean) => {
  const color = typeColors[field.type];
  const key = `${field.type}-${isSelected}`;
  if (iconCache.has(key)) return iconCache.get(key)!;
  const size = isSelected ? 44 : 34;
  const icon = L.divIcon({
    html: markerSvg(color, isSelected, field.type),
    className: "",
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
  iconCache.set(key, icon);
  return icon;
};

const tileUrls: Record<MapStyle, { url: string; attribution: string; subdomains?: string }> = {
  standard: {
    url: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
    attribution: "&copy; CARTO",
  },
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "&copy; Esri",
  },
  terrain: {
    url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenTopoMap (CC-BY-SA)",
    subdomains: "abc",
  },
};

// ─── Color helpers ────────────────────────────────────────────────────────────

// Returns a fill color based on NDVI value (0–1 scale)
// Low NDVI = red (stressed), mid = yellow, high = green (healthy)
const ndviToColor = (ndvi: number): string => {
  if (ndvi >= 0.45) return "#22c55e"; // green  – healthy
  if (ndvi >= 0.30) return "#eab308"; // yellow – moderate
  return "#ef4444";                   // red    – stressed
};

// Returns fill color based on active layer and state's CSV data
const getStateFillColor = (stateName: string, activeLayer: LayerKey): string => {
  const data = csvStateData[stateName];
  if (!data) return "#94a3b8"; // grey for states not in CSV

  switch (activeLayer) {
    case "ndvi":
    case "healthy":
      return ndviToColor(data.ndvi);

    case "rainfall":
      // rainfall: low = red, high = blue
      if (data.rainfall >= 7) return "#0ea5e9";
      if (data.rainfall >= 5) return "#38bdf8";
      return "#ef4444";

    case "temperature":
      // temperature: cool = blue, hot = red
      if (data.temperature >= 29) return "#ef4444";
      if (data.temperature >= 27) return "#eab308";
      return "#0ea5e9";

    case "risk":
      // Drought/Flood risk heatmap: lower NDVI/NDWI indicates higher drought risk
      if (data.ndvi < 0.30 || data.ndwi < -0.40) return "#ef4444"; // High Risk (Red)
      if (data.ndvi < 0.45) return "#eab308"; // Moderate Risk (Yellow)
      return "#3b82f6"; // Low Risk (Blue)

    case "irrigation":
      // water stress via NDWI — less negative = more water
      if (data.ndwi > -0.35) return "#0ea5e9";
      if (data.ndwi > -0.45) return "#eab308";
      return "#ef4444";
      
    case "yield":
      // Yield prediction contours: scale based on NDVI
      if (data.ndvi >= 0.45) return "#166534"; // High yield (Dark Green)
      if (data.ndvi >= 0.30) return "#4ade80"; // Moderate yield (Light Green)
      return "#facc15"; // Low yield (Yellow)

    default:
      return ndviToColor(data.ndvi);
  }
};

// ─── Map sub-components ───────────────────────────────────────────────────────

function MapBoundsEnforcer() {
  const map = useMap();
  useEffect(() => {
    const nigeriaBounds = L.latLngBounds(NIGERIA_BOUNDS);
    const fitNigeria = () => {
      const boundsZoom = map.getBoundsZoom(nigeriaBounds, false);
      map.setMinZoom(boundsZoom); // Allow zooming out to see whole country
      map.setMaxBounds(nigeriaBounds.pad(0)); // Strict bounds to Nigeria map only
      // Zoom in 100% tighter into the center
      map.setView(nigeriaBounds.getCenter(), boundsZoom + 1.5);
    };
    fitNigeria();
    map.on("resize", fitNigeria);
    return () => { map.off("resize", fitNigeria); };
  }, [map]);
  return null;
}

function FlyToLocation({ position, zoom }: { position: [number, number] | null; zoom?: number }) {
  const map = useMap();
  const prevRef = useRef<string | null>(null);
  useEffect(() => {
    if (!position) return;
    const key = `${position[0]},${position[1]},${zoom}`;
    if (prevRef.current === key) return;
    prevRef.current = key;
    map.flyTo(position, zoom || 10, { duration: 1.2, easeLinearity: 0.3 });
  }, [position, zoom, map]);
  return null;
}

function MapStyleUpdater({ mapStyle }: { mapStyle: MapStyle }) {
  const map = useMap();
  const tileRef = useRef<L.TileLayer | null>(null);
  useEffect(() => {
    if (tileRef.current) map.removeLayer(tileRef.current);
    const cfg = tileUrls[mapStyle];
    const tile = L.tileLayer(cfg.url, {
      maxZoom: 18,
      updateWhenIdle: true,
      updateWhenZooming: false,
      keepBuffer: 3,
      subdomains: cfg.subdomains ?? "abc",
      attribution: cfg.attribution,
    });
    tile.addTo(map);
    tileRef.current = tile;
    return () => { if (tileRef.current) map.removeLayer(tileRef.current); };
  }, [mapStyle, map]);
  return null;
}

function MapClickHandler({ onClick }: { onClick: () => void }) {
  const map = useMap();
  useEffect(() => {
    const handler = () => onClick();
    map.on("click", handler);
    return () => { map.off("click", handler); };
  }, [map, onClick]);
  return null;
}

function MapZoomController() {
  const map = useMap();
  useEffect(() => {
    const zoomIn = () => map.zoomIn();
    const zoomOut = () => map.zoomOut();
    window.addEventListener("map:zoomIn", zoomIn);
    window.addEventListener("map:zoomOut", zoomOut);
    return () => {
      window.removeEventListener("map:zoomIn", zoomIn);
      window.removeEventListener("map:zoomOut", zoomOut);
    };
  }, [map]);
  return null;
}

// ─── Main component ───────────────────────────────────────────────────────────

const NigeriaMap = ({
  activeLayer,
  activeFeature,
  csvMetrics,
  mapStyle,
  mapView,
  selectedField,
  selectedState,
  flyTo,
  flyToZoom,
  riceFields,
  rawRecords,
  onFieldClick,
  onMapClick,
  onStateClick,
}: NigeriaMapProps) => {
  // ─── Dynamic Field position lookup by ID ────────────────────────────────────
  const fieldPositionById = useMemo(() => {
    const map = new Map<string, [number, number]>();
    for (const f of riceFields) map.set(f.id, f.position);
    return map;
  }, [riceFields]);
  // Load the GeoJSON file from /public/nigeria_states.geojson
  const [geoData, setGeoData] = useState<GeoJSON.FeatureCollection | null>(null);

  useEffect(() => {
    fetch("/nigeria_states.geojson")
      .then((r) => r.json())
      .then(setGeoData)
      .catch((err) => console.error("Failed to load nigeria_states.geojson:", err));
  }, []);

  const layerMeta = useMemo(() => LAYERS.find((l) => l.key === activeLayer)!, [activeLayer]);

  // Only render the 6 states that have real CSV data
  const filteredGeoData = useMemo(() => {
    if (!geoData) return null;
    return {
      ...geoData,
      features: geoData.features.filter(
        (f: any) => (f?.properties?.shapeName || f?.properties?.name) in csvStateData
      ),
    } as GeoJSON.FeatureCollection;
  }, [geoData]);



  const visibleFields = useMemo(() => {
    // Default ("all"): show all rice fields
    if (activeFeature === "all") return riceFields;
    // Feature selected: filter by both layer type AND CSV metrics filter
    const allow = layerToFieldTypes[activeLayer] || ["healthy"];
    return riceFields.filter((f) => allow.includes(f.type));
  }, [activeLayer, activeFeature, csvMetrics, riceFields]);

  // Style each state polygon based on real CSV data + active layer.
  // Design intent: the heatmap colour is used as a BORDER stroke so the actual
  // state shape is highlighted against the base map, not buried under an opaque fill.
  const stateStyle = useMemo(() => (feature: any): L.PathOptions => {
    const stateName: string = (feature?.properties?.shapeName || feature?.properties?.name) ?? "";
    const metrics = csvMetrics?.[stateName];
    const isInDataset = !!metrics;

    if (!isInDataset) {
      // Non-data states: very faint grey outline, almost invisible fill
      return {
        color: "#94a3b8",
        weight: 0.8,
        opacity: 0.4,
        fillColor: "#94a3b8",
        fillOpacity: 0.03,
      };
    }

    let borderColor = getStateFillColor(stateName, activeLayer);

    // Override colour based on advanced features
    if (metrics) {
      if (activeFeature === "temp_stress") {
        const t = metrics.avgTemperature;
        borderColor = t > 32 ? "#ef4444" : t > 30 ? "#f97316" : t > 28 ? "#fcd34d" : "#3b82f6";
      } else if (activeFeature === "rainfall_patterns") {
        const p = metrics.avgPrecipitation;
        borderColor = p > 6 ? "#1e3a8a" : p > 4.5 ? "#3b82f6" : p > 3.5 ? "#fcd34d" : "#ef4444";
      } else if (activeFeature === "historical") {
        const diff = metrics.ndviDiff;
        borderColor = diff > 0.02 ? "#22c55e" : diff < -0.02 ? "#ef4444" : "#facc15";
      } else if (activeFeature === "growth_trends") {
        const trend = metrics.ndviTrend;
        borderColor = trend > 0.01 ? "#22c55e" : trend < -0.01 ? "#ef4444" : "#94a3b8";
      } else if (activeFeature === "irrigation_junctions") {
        const ndwi = metrics.latestNdwi;
        borderColor = ndwi < -0.45 ? "#ef4444" : ndwi < -0.35 ? "#f97316" : "#0ea5e9";
      }
    }

    const inFarmView = mapView === "farm";
    return {
      // Keep a very subtle tint of the border colour as a fill so the
      // state area is gently distinguished — but the map stays fully visible.
      fillColor: borderColor,
      fillOpacity: inFarmView ? 0.06 : 0.12,
      // The prominent border is the main visual element
      color: borderColor,
      weight: inFarmView ? 1.5 : 3.5,
      opacity: inFarmView ? 0.6 : 1.0,
      // Dash array for custom_zones draw mode to look different
      ...(activeFeature === "custom_zones" ? { dashArray: "6 4" } : {}),
    };
  }, [activeLayer, activeFeature, csvMetrics, mapView]);

  const onEachFeature = useMemo(() => (feature: any, layer: L.Layer) => {
    layer.on({
      click: (e) => {
        L.DomEvent.stopPropagation(e);
        const stateName = feature?.properties?.shapeName || feature?.properties?.name;
        if (stateName && onStateClick) {
          onStateClick(stateName);
        } else {
          onMapClick();
        }
      }
    });
  }, [onStateClick, onMapClick]);

  const layerStyle = useMemo(
    () => (feature: any) => {
      const v = feature?.properties?.value ?? 0.5;
      return {
        color: layerMeta.color,
        weight: feature?.geometry?.type === "LineString" ? 3 : 0.4,
        opacity: feature?.geometry?.type === "LineString" ? 0.9 : 0.3,
        fillColor: layerMeta.color,
        fillOpacity: 0.15 + v * 0.45,
      };
    },
    [layerMeta]
  );

  const pointToLayer = useMemo(
    () => (_feat: any, latlng: L.LatLng) =>
      L.circleMarker(latlng, {
        radius: 4,
        color: "white",
        weight: 1.5,
        fillColor: layerMeta.color,
        fillOpacity: 1,
      }),
    [layerMeta]
  );

  return (
    <MapContainer
      center={NIGERIA_CENTER}
      zoom={6.5}
      className="w-full h-full"
      zoomControl={false}
      attributionControl={false}
      maxBounds={L.latLngBounds(NIGERIA_BOUNDS).pad(0)}
      maxBoundsViscosity={1.0}
      minZoom={5}
      maxZoom={18}
      zoomSnap={0.25}
      zoomDelta={0.5}
      wheelDebounceTime={80}
      wheelPxPerZoomLevel={120}
      preferCanvas={true}
      inertia={true}
      inertiaDeceleration={3000}
    >
      <MapBoundsEnforcer />
      <MapStyleUpdater mapStyle={mapStyle} />
      <FlyToLocation position={flyTo} zoom={flyToZoom} />
      <MapClickHandler onClick={onMapClick} />
      <MapZoomController />
      <MapLegend activeFeature={activeFeature} />

      {/* ── State Polygons ── */}
      {filteredGeoData && !["heat_signatures", "rainfall_events", "water_stress"].includes(activeFeature) && (
        <GeoJSON
          key={`states-${activeLayer}-${activeFeature}-${mapView}`}
          data={filteredGeoData}
          style={stateStyle}
          onEachFeature={onEachFeature}
        />
      )}

      {/* ── Advanced Visualization Layers ── */}
      {filteredGeoData && activeFeature === "heat_signatures" && (
        <HeatSignatureLayer geoData={filteredGeoData as any} metrics={csvMetrics} />
      )}
      {filteredGeoData && activeFeature === "rainfall_events" && (
        <TemporalScatterLayer geoData={filteredGeoData as any} metrics={csvMetrics} rawRecords={rawRecords} />
      )}
      {filteredGeoData && activeFeature === "water_stress" && (
        <WaterStressChoropleth geoData={filteredGeoData as any} metrics={csvMetrics} />
      )}

      {/* ── Advanced Raster Overlays ── */}
      <AdvancedRasterLayer geoData={filteredGeoData} activeFeature={activeFeature} />

      {/* ── Irrigation Junctions: water-network polylines ── */}
      {activeFeature === "irrigation_junctions" && mockIrrigationJunctions.map((junc) =>
        junc.connectedFarms.map((farmId) => {
          const farmPos = fieldPositionById.get(farmId);
          if (!farmPos) return null;
          const lineColor = junc.status === "Good" ? "#0ea5e9" : junc.status === "Low" ? "#f97316" : "#ef4444";
          return (
            <Polyline
              key={`line-${junc.id}-${farmId}`}
              positions={[junc.position, farmPos]}
              pathOptions={{ color: lineColor, weight: 2, opacity: 0.7, dashArray: "5 4" }}
            />
          );
        })
      )}

      {/* ── Irrigation Junction markers ── */}
      {activeFeature === "irrigation_junctions" && mockIrrigationJunctions.map((junc) => (
        <Marker
          key={junc.id}
          position={junc.position}
          icon={getJunctionIcon(junc.status)}
        >
          <Popup>
            <div style={{ fontFamily: "Inter,sans-serif", minWidth: 190 }}>
              <p style={{ fontWeight: 700, fontSize: 13, marginBottom: 6, color: "#1e293b" }}>
                💧 {junc.name}
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                <div style={{ background: "#f1f5f9", borderRadius: 8, padding: "6px 8px" }}>
                  <div style={{ fontSize: 9, color: "#64748b", fontWeight: 600, textTransform: "uppercase" }}>Status</div>
                  <div style={{
                    fontSize: 13, fontWeight: 700,
                    color: junc.status === "Good" ? "#0ea5e9" : junc.status === "Low" ? "#f97316" : "#ef4444"
                  }}>{junc.status}</div>
                </div>
                <div style={{ background: "#f1f5f9", borderRadius: 8, padding: "6px 8px" }}>
                  <div style={{ fontSize: 9, color: "#64748b", fontWeight: 600, textTransform: "uppercase" }}>Updated</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "#1e293b" }}>{junc.lastUpdated}</div>
                </div>
              </div>
              <p style={{ fontSize: 10, color: "#64748b", marginTop: 6 }}>
                Connected to {junc.connectedFarms.length} farm{junc.connectedFarms.length > 1 ? "s" : ""} (ID: {junc.id})
              </p>
            </div>
          </Popup>
        </Marker>
      ))}

      {/* ── Rice field markers (Only in Farm View) ── */}
      {mapView === "farm" && visibleFields.map((field) => (
        <Marker
          key={field.id}
          position={field.position}
          icon={getMarkerIcon(field, selectedField?.id === field.id)}
          eventHandlers={{
            click: (e) => {
              (e.originalEvent as MouseEvent).stopPropagation?.();
              onFieldClick(field);
            },
          }}
        >
          <Tooltip direction="right" offset={[10, 0]} opacity={1} permanent={true} className="bg-background/90 text-foreground font-semibold border-border/50 shadow-sm px-1.5 py-0.5 text-[10px]">
            {field.name}
          </Tooltip>
        </Marker>
      ))}

      {/* ── Custom Farm Zones draw control ── */}
      {activeFeature === "custom_zones" && (
        <MapDrawControl csvMetrics={csvMetrics} />
      )}
    </MapContainer>
  );
};

export default NigeriaMap;