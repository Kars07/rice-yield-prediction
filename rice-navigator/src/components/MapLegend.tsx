import { motion, AnimatePresence } from "framer-motion";
import type { FeatureKey } from "@/data/csvProcessor";

interface Props {
  activeFeature: FeatureKey;
}

const LEGENDS: Partial<Record<FeatureKey, { title: string; min: string; max: string; gradient: string; colors: string[] }>> = {
  temp_stress: {
    title: "Temperature Stress",
    min: "Cooler",
    max: "High Stress (>35°C)",
    gradient: "linear-gradient(to right, #3b82f6, #fcd34d, #f97316, #ef4444)",
    colors: ["#3b82f6", "#fcd34d", "#f97316", "#ef4444"],
  },
  rainfall_patterns: {
    title: "Rainfall Patterns",
    min: "Deficit",
    max: "Surplus",
    gradient: "linear-gradient(to right, #ef4444, #fcd34d, #3b82f6, #1e3a8a)",
    colors: ["#ef4444", "#fcd34d", "#3b82f6", "#1e3a8a"],
  },
  historical: {
    title: "Yield Diff (vs 2024)",
    min: "Worse",
    max: "Better",
    gradient: "linear-gradient(to right, #ef4444, #facc15, #22c55e)",
    colors: ["#ef4444", "#facc15", "#22c55e"],
  },
  irrigation_junctions: {
    title: "Irrigation Status",
    min: "Critical",
    max: "Good",
    gradient: "linear-gradient(to right, #ef4444, #f97316, #0ea5e9)",
    colors: ["#ef4444", "#f97316", "#0ea5e9"],
  },
  custom_zones: {
    title: "Draw Mode Active",
    min: "Draw a polygon",
    max: "See zonal stats",
    gradient: "linear-gradient(to right, #c4b5fd, #8b5cf6)",
    colors: ["#c4b5fd", "#8b5cf6"],
  },
};

const MapLegend = ({ activeFeature }: Props) => {
  const legend = LEGENDS[activeFeature];

  return (
    <AnimatePresence>
      {legend && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          className="absolute bottom-[90px] md:bottom-6 right-3 md:right-6 z-[1000] bg-card rounded-xl p-3 shadow-xl border border-border/40 pointer-events-auto"
        >
          <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-2">
            {legend.title}
          </p>
          <div className="w-36 h-2.5 rounded-full mb-1.5" style={{ background: legend.gradient }} />
          <div className="flex justify-between text-[9px] font-semibold text-foreground">
            <span>{legend.min}</span>
            <span>{legend.max}</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default MapLegend;
