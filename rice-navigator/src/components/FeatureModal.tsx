import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Leaf,
  TrendingUp,
  Activity,
  BarChart2,
  Thermometer,
  History,
  AlertTriangle,
  Sprout,
  Droplets,
  BookmarkPlus,
  PenLine,
  Cpu,
  Lightbulb,
  Star,
  Flame,
  CloudRain
} from "lucide-react";
import type { FeatureKey } from "@/data/csvProcessor";

// ─── Feature catalogue ────────────────────────────────────────────────────────

interface FeatureItem {
  key: FeatureKey | "bookmark" | "custom_zones" | "ensemble" | "recommendations";
  label: string;
  icon: React.ElementType;
  color: string;
  fromCSV: boolean;   // true = backed by real data, false = UI-only
  badge?: string;
}

const CATEGORIES: { title: string; features: FeatureItem[] }[] = [
  {
    title: "Crop Health",
    features: [
      { key: "healthy_fields",  label: "Healthy Fields",     icon: Leaf,       color: "#22c55e", fromCSV: true  },
      { key: "ndvi_vigor",      label: "NDVI / EVI Vigor",   icon: TrendingUp,  color: "#eab308", fromCSV: true  },
      { key: "phenology",       label: "Phenology Stages",   icon: Sprout,      color: "#84cc16", fromCSV: true  },
      { key: "growth_trends",   label: "Growth Rate Trends", icon: Activity,    color: "#10b981", fromCSV: true  },
    ],
  },
  {
    title: "Climate & Risk",
    features: [
      { key: "temp_stress",  label: "Temperature Stress",     icon: Thermometer, color: "#f97316", fromCSV: true  },
      { key: "rainfall_patterns", label: "Rainfall Patterns", icon: Droplets,    color: "#0ea5e9", fromCSV: true  },
      { key: "historical",   label: "Historical Comparison",  icon: History,     color: "#3b82f6", fromCSV: true  },
      { key: "all",          label: "Pest Hotspots",          icon: AlertTriangle, color: "#ef4444", fromCSV: true,  },
    ],
  },
  {
    title: "Infrastructure",
    features: [
      { key: "irrigation_junctions", label: "Irrigation Junctions", icon: Droplets,  color: "#3b82f6", fromCSV: false,  },
      { key: "custom_zones",         label: "Custom Farm Zones",    icon: PenLine,   color: "#8b5cf6", fromCSV: false, },
    ],
  },
  {
    title: "Analytics",
    features: [
      { key: "ensemble",         label: "Ensemble Breakdown",       icon: Cpu,       color: "#06b6d4", fromCSV: true  },
      { key: "ndvi_vigor",       label: "Feature Importance",       icon: BarChart2, color: "#a855f7", fromCSV: true  },
      { key: "recommendations",  label: "Actionable Recommendations",icon: Lightbulb, color: "#f59e0b", fromCSV: true  },
    ],
  },
  {
    title: "Advanced Maps",
    features: [
      { key: "heat_signatures",  label: "Heat Signatures",          icon: Flame,     color: "#f97316", fromCSV: true  },
      { key: "rainfall_events",  label: "Rainfall Events",          icon: CloudRain, color: "#0ea5e9", fromCSV: true  },
      { key: "water_stress",     label: "Water Stress",             icon: Droplets,  color: "#3b82f6", fromCSV: true  },
    ],
  },
];

interface Props {
  open: boolean;
  activeFeature: FeatureKey;
  onSelect: (key: FeatureKey) => void;
  onClose: () => void;
}

const FeatureModal = ({ open, activeFeature, onSelect, onClose }: Props) => {
  const handleSelect = (key: string) => {
    const validKeys: FeatureKey[] = [
      "all", "healthy_fields", "ndvi_vigor", "phenology",
      "growth_trends", "temp_stress", "historical", "irrigation_junctions", "custom_zones",
      "heat_signatures", "rainfall_events", "water_stress"
    ];
    const safeKey = validKeys.includes(key as FeatureKey) ? (key as FeatureKey) : "all";
    onSelect(safeKey);
    onClose();
  };

  return createPortal(
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[1100]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Sheet */}
          <motion.div
            className="fixed bottom-0 left-0 right-0 z-[1101] bg-card rounded-t-3xl shadow-2xl border-t border-border/40 max-h-[85vh] overflow-hidden"
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 320 }}
          >
            {/* Handle */}
            <div className="flex justify-center pt-3 pb-1">
              <div className="w-10 h-1 rounded-full bg-muted-foreground/30" />
            </div>

            {/* Header */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-border/30">
              <div>
                <h2 className="text-base font-bold text-foreground">All Features</h2>
                <p className="text-[11px] text-muted-foreground">Tap any chip to filter the map</p>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-full bg-muted flex items-center justify-center"
              >
                <X className="w-4 h-4 text-muted-foreground" />
              </button>
            </div>

            {/* Scrollable content */}
            <div className="overflow-y-auto pb-10 no-scrollbar">
              {CATEGORIES.map((cat) => (
                <div key={cat.title} className="px-5 pt-5">
                  <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3">
                    {cat.title}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {cat.features.map((feat, i) => {
                      const isActive = activeFeature === feat.key;
                      const Icon = feat.icon;
                      return (
                        <motion.button
                          key={`${feat.key}-${i}`}
                          whileTap={{ scale: 0.95 }}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.04 }}
                          onClick={() => handleSelect(feat.key)}
                          className={`flex items-center gap-2 px-3.5 py-2 rounded-full text-xs font-semibold border transition-all duration-200 ${
                            isActive
                              ? "border-transparent text-white shadow-lg"
                              : "bg-muted/40 border-border/40 text-foreground hover:border-primary/30"
                          }`}
                          style={
                            isActive
                              ? { backgroundColor: feat.color, borderColor: feat.color }
                              : {}
                          }
                        >
                          <Icon
                            className="w-3 h-3 shrink-0"
                            style={{ color: isActive ? "white" : feat.color }}
                          />
                          {feat.label}
                          {feat.badge && !isActive && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
                              {feat.badge}
                            </span>
                          )}
                          {!feat.fromCSV && !isActive && (
                            <Star className="w-2.5 h-2.5 text-muted-foreground/50" />
                          )}
                        </motion.button>
                      );
                    })}
                  </div>
                </div>
              ))}

             
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
};

// Need to import createPortal
import { createPortal } from "react-dom";

export default FeatureModal;
export { CATEGORIES };
export type { FeatureItem };
