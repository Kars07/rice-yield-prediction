import { useState } from "react";
import { motion } from "framer-motion";
import {
  Leaf, TrendingUp, Activity, Thermometer, History, ChevronDown, Droplets, PenLine, Map, Tractor, Flame, CloudRain
} from "lucide-react";
import type { FeatureKey } from "@/data/csvProcessor";
import FeatureModal from "@/components/FeatureModal";

interface Props {
  mapView: "state" | "farm";
  onChangeView: (view: "state" | "farm") => void;
  activeFeature: FeatureKey;
  onChange: (key: FeatureKey) => void;
}

// Quick-access chips shown in the horizontal bar
const QUICK_CHIPS: { key: FeatureKey; label: string; icon: React.ElementType; color: string }[] = [
  { key: "heat_signatures",    label: "Heat Signatures", icon: Flame,       color: "#f97316" },
  { key: "rainfall_events",    label: "Rainfall Events", icon: CloudRain,   color: "#0ea5e9" },
  { key: "water_stress",       label: "Water Stress",    icon: Droplets,    color: "#3b82f6" },
  { key: "healthy_fields",     label: "Healthy",      icon: Leaf,        color: "#22c55e" },
  { key: "ndvi_vigor",         label: "NDVI Vigor",   icon: TrendingUp,  color: "#eab308" },
  { key: "custom_zones",       label: "Draw Zone",    icon: PenLine,     color: "#8b5cf6" },
];

const FilterChips = ({ mapView, onChangeView, activeFeature, onChange }: Props) => {
  const [modalOpen, setModalOpen] = useState(false);

  return (
    <>
      <motion.div
        className="fixed top-[75px] left-0 right-0 md:top-4 md:left-[37rem] md:right-4 md:pt-1.5 z-[999] px-4 md:px-0"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
      >
        <div className="flex gap-2 overflow-x-auto no-scrollbar py-1.5 items-center">

          {/* View toggle chips */}
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={() => onChangeView("state")}
            className={`shrink-0 flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold border transition-all duration-200 ${
              mapView === "state"
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "bg-card text-foreground border-border/40 hover:border-primary/30"
            }`}
          >
            <Map className="w-3.5 h-3.5" />
            States
          </motion.button>

          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={() => onChangeView("farm")}
            className={`shrink-0 flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold border transition-all duration-200 ${
              mapView === "farm"
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "bg-card text-foreground border-border/40 hover:border-primary/30"
            }`}
          >
            <Tractor className="w-3.5 h-3.5" />
            Farms
          </motion.button>

          {/* Quick-access chips */}
          {QUICK_CHIPS.map((chip, i) => {
            const isActive = activeFeature === chip.key;
            const Icon = chip.icon;
            return (
              <motion.button
                key={chip.key}
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => onChange(chip.key)}
                className={`shrink-0 px-4 py-2 rounded-full text-xs font-semibold border transition-all duration-200 flex items-center gap-2 ${
                  isActive
                    ? "text-white border-transparent shadow-md"
                    : "bg-card text-foreground border-border/40 hover:border-primary/30"
                }`}
                style={isActive ? { backgroundColor: chip.color, borderColor: chip.color } : {}}
              >
                <Icon
                  className="w-3 h-3"
                  style={{ color: isActive ? "white" : chip.color }}
                />
                {chip.label}
              </motion.button>
            );
          })}

          {/* More button → opens FeatureModal */}
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={() => setModalOpen(true)}
            className={`shrink-0 px-4 py-2 rounded-full text-xs font-semibold border flex items-center gap-1.5 transition-all duration-200 ${
              modalOpen
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "bg-card text-foreground border-border/40 hover:border-primary/30"
            }`}
          >
            <ChevronDown className="w-3 h-3" />
            More
          </motion.button>

        </div>
      </motion.div>

      {/* Slide-up feature modal */}
      <FeatureModal
        open={modalOpen}
        activeFeature={activeFeature}
        onSelect={(key) => {
          onChange(key);
          setModalOpen(false);
        }}
        onClose={() => setModalOpen(false)}
      />
    </>
  );
};

export default FilterChips;