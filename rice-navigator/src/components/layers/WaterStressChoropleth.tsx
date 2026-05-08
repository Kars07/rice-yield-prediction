import { GeoJSON, Polyline, Tooltip } from "react-leaflet";
import { type StateMetrics } from "@/data/csvProcessor";

interface Props {
  geoData: GeoJSON.FeatureCollection;
  metrics: Record<string, StateMetrics> | null;
}

export const WaterStressChoropleth = ({ geoData, metrics }: Props) => {

  // 1. Compute a water intensity score per state
  const getScore = (stateName: string): number => {
    if (!metrics || !metrics[stateName]) return -1;
    const ndwi = metrics[stateName].latestNdwi;   // e.g. -0.5 to 0.1
    const precip = metrics[stateName].avgPrecipitation; // mm
    // Combine: normalise both to 0-1 and average
    const ndwiNorm = Math.min(1, Math.max(0, (ndwi + 0.5) / 0.6));
    const precipNorm = Math.min(1, Math.max(0, (precip - 1) / 9));
    return (ndwiNorm * 0.5 + precipNorm * 0.5); // 0 = very dry, 1 = very wet
  };

  // 2. Map score to a 5-step monochromatic Blue scheme
  const getColor = (score: number): string => {
    if (score < 0)    return "#e5e7eb"; // No data — light grey
    if (score < 0.20) return "#dbeafe"; // Lightest blue
    if (score < 0.40) return "#93c5fd"; // Light blue
    if (score < 0.60) return "#3b82f6"; // Medium blue
    if (score < 0.80) return "#1d4ed8"; // Dark blue
    return "#1e3a8a";                   // Navy
  };

  // 3. Style function — clear borders so state lines are visible
  const style = (feature: any) => {
    const score = getScore(feature.properties.shapeName);
    return {
      fillColor: getColor(score),
      fillOpacity: 0.72,       // Reduced from 0.85 — map texture shows through
      color: "#ffffff",        // WHITE border → crisp contrast against blue fill
      weight: 2,               // Thicker than before
      opacity: 0.9,
    };
  };

  return (
    <>
      <GeoJSON
        key={`choropleth-${JSON.stringify(metrics ? Object.keys(metrics) : [])}`}
        data={geoData}
        style={style}
      />
      <RiverOverlays />
      <WaterStressLegend />
    </>
  );
};

// Major Nigerian rivers with accurate simplified coordinates
const RiverOverlays = () => {
  // Niger River: flows S through Kebbi → Niger → Kogi → Delta
  const nigerRiver: [number, number][] = [
    [13.5, 3.9], [12.0, 4.2], [10.0, 4.5],
    [8.5, 5.2], [7.8, 6.5], [7.2, 6.7],
    [6.4, 6.4], [5.5, 6.2],
  ];
  // Benue River: flows W from Taraba → Benue → Kogi confluence
  const benueRiver: [number, number][] = [
    [8.3, 13.5], [7.9, 11.5], [7.8, 9.8],
    [7.7, 8.3], [7.5, 7.0], [7.2, 6.7],
  ];

  return (
    <>
      <Polyline
        positions={nigerRiver}
        pathOptions={{ color: "#0ea5e9", weight: 3, opacity: 0.85 }}
      >
        <Tooltip
          permanent
          direction="top"
          offset={[0, -4]}
          className="!bg-transparent !border-0 !shadow-none !text-sky-700 !font-bold !text-[10px]"
        >
          Niger River
        </Tooltip>
      </Polyline>
      <Polyline
        positions={benueRiver}
        pathOptions={{ color: "#38bdf8", weight: 2.5, opacity: 0.8 }}
      >
        <Tooltip
          permanent
          direction="top"
          offset={[0, -4]}
          className="!bg-transparent !border-0 !shadow-none !text-sky-600 !font-bold !text-[10px]"
        >
          Benue River
        </Tooltip>
      </Polyline>
    </>
  );
};

const WaterStressLegend = () => (
  <div
    className="absolute z-[1000] bg-white/95 backdrop-blur-sm px-4 py-3 rounded-xl shadow-2xl border border-gray-200"
    style={{ bottom: "90px", left: "50%", transform: "translateX(-50%)" }}
  >
    <div className="text-[11px] font-bold mb-1 text-gray-700 text-center tracking-widest uppercase">
      Blue Water Intensity 2023
    </div>
    <div className="text-[9px] text-gray-400 text-center mb-2">
      Combined NDWI + Precipitation index
    </div>

    {/* Continuous gradient bar */}
    <div
      className="h-3 w-64 rounded-full mb-1"
      style={{
        background: "linear-gradient(to right, #dbeafe, #93c5fd, #3b82f6, #1d4ed8, #1e3a8a)",
      }}
    />

    {/* Tick labels */}
    <div className="flex justify-between text-[9px] font-semibold text-gray-500">
      <span>Very Low</span>
      <span>Low</span>
      <span>Medium</span>
      <span>High</span>
      <span>Very High</span>
    </div>

    {/* No-data swatch */}
    <div className="flex items-center gap-2 mt-2 text-[9px] font-semibold text-gray-500">
      <div className="w-4 h-3 rounded bg-gray-200 border border-gray-300" />
      No data
    </div>
  </div>
);
