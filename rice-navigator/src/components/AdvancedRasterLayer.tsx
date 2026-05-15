import { useEffect, useState, useMemo } from 'react';
import { useMap, GeoJSON } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet.heat';
import { featureCollection, booleanPointInPolygon, center, squareGrid } from '@turf/turf';
import { type FeatureKey } from '@/data/csvProcessor';

interface Props {
  geoData: GeoJSON.FeatureCollection | null;
  activeFeature: FeatureKey;
}

// Ensure the types for leaflet.heat are accepted by TS
declare module 'leaflet' {
  function heatLayer(latlngs: any[], options: any): any;
}

const AdvancedRasterLayer = ({ geoData, activeFeature }: Props) => {
  const map = useMap();
  const [gridData, setGridData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [heatmapLayer, setHeatmapLayer] = useState<any>(null);

  useEffect(() => {
    if (!geoData) return;

    // We only generate Advanced Rasters for certain features that need "Scientific" representation
    const needsGrid = ["ndvi_vigor", "risk_heatmap"].includes(activeFeature);
    const needsHeatmap = ["temp_stress"].includes(activeFeature);

    // Clean up previous layers
    setGridData(null);
    if (heatmapLayer) {
      map.removeLayer(heatmapLayer);
      setHeatmapLayer(null);
    }

    if (!needsGrid && !needsHeatmap) return;

    if (needsHeatmap) {
      // Generate mock heat points across the states
      const heatPoints: [number, number, number][] = [];
      geoData.features.forEach((feature: any) => {
        const bounds = L.geoJSON(feature).getBounds();
        const numPoints = 150; // Points per state
        for (let i = 0; i < numPoints; i++) {
          const lat = bounds.getSouth() + Math.random() * (bounds.getNorth() - bounds.getSouth());
          const lng = bounds.getWest() + Math.random() * (bounds.getEast() - bounds.getWest());
          // Intensity is random but clustered
          const intensity = Math.random() > 0.8 ? 1.0 : Math.random() * 0.5;
          heatPoints.push([lat, lng, intensity]);
        }
      });

      const hl = L.heatLayer(heatPoints, {
        radius: 45,
        blur: 35,
        maxZoom: 10,
        max: 1.0,
        gradient: {
          0.0: '#ffff00', // Yellow
          0.4: '#ff8c00', // Orange
          0.7: '#ff0000', // Red
          1.0: '#8b0000', // Dark Red
        }
      });
      hl.addTo(map);
      setHeatmapLayer(hl);
    }

    if (needsGrid) {
      // Generate a pixelated square grid masking the states
      const allCells: any[] = [];
      const cellSize = 30; // 30km cell size to dramatically reduce number of cells and prevent freezing
      
      geoData.features.forEach((feature: any) => {
        if (!feature.geometry) return;
        const bounds = L.geoJSON(feature).getBounds();
        const bbox = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
        
        try {
          const grid = squareGrid(bbox, cellSize, { units: 'kilometers' });
          
          grid.features.forEach((cell) => {
            // Use booleanPointInPolygon on the cell's center for ultra-fast point-in-polygon checks
            // instead of complex boolean intersection.
            const cellCenter = center(cell);
            if (booleanPointInPolygon(cellCenter, feature)) {
              let val = Math.random();
              
              if (activeFeature === "yield_prediction") {
                val = (Math.random() + Math.random() + Math.random()) / 3;
              } else if (activeFeature === "ndvi_vigor") {
                val = Math.random();
              } else if (activeFeature === "risk_heatmap") {
                val = Math.random();
              }
              
              cell.properties = { value: val };
              allCells.push(cell);
            }
          });
        } catch (e) {
          console.warn("Grid gen failed for a state", e);
        }
      });

      setGridData(featureCollection(allCells));
    }

    return () => {
      if (heatmapLayer) map.removeLayer(heatmapLayer);
    };
  }, [geoData, activeFeature, map]);

  const style = useMemo(() => (feature: any) => {
    const val = feature.properties.value;
    let color = "#000000";

    if (activeFeature === "yield_prediction") {
      // <2000 (red), 2000-3000 (orange), 3000-4000 (yellow), 4000-5000 (light green), >5000 (dark green)
      if (val < 0.2) color = "#d7191c";
      else if (val < 0.4) color = "#fdae61";
      else if (val < 0.6) color = "#ffffbf";
      else if (val < 0.8) color = "#a6d96a";
      else color = "#1a9641";
    } else if (activeFeature === "ndvi_vigor") {
      // RGB terrain colors (blue, green, yellow, red, dark red)
      if (val < 0.2) color = "#2c7bb6";
      else if (val < 0.4) color = "#abd9e9";
      else if (val < 0.6) color = "#ffffbf";
      else if (val < 0.8) color = "#fdae61";
      else color = "#d7191c";
    } else if (activeFeature === "risk_heatmap") {
      // Water stress (Blue Intensity)
      if (val < 0.2) color = "#c6dbef";
      else if (val < 0.4) color = "#9ecae1";
      else if (val < 0.6) color = "#6baed6";
      else if (val < 0.8) color = "#3182bd";
      else color = "#08519c";
    }

    return {
      fillColor: color,
      fillOpacity: 0.85,
      weight: 0, // No border for pixels
      color: "transparent",
      opacity: 0,
    };
  }, [activeFeature]);

  if (gridData) {
    return <GeoJSON key={`grid-${activeFeature}`} data={gridData} style={style} />;
  }

  return null;
};

export default AdvancedRasterLayer;
