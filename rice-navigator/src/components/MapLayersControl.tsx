import { useState } from "react";
import { Layers, X, Mountain, Map as MapIcon, Satellite, Plus, Minus, TrendingUp, AlertTriangle, Sprout, Flame, CloudRain, Droplets } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import type { FeatureKey } from "@/data/csvProcessor";

export type MapStyle = "standard" | "satellite" | "terrain";

interface Props {
  mapStyle: MapStyle;
  activeFeature: FeatureKey;
  onStyleChange: (s: MapStyle) => void;
  onFeatureChange: (f: FeatureKey) => void;
}

const styles: { key: MapStyle; label: string; Icon: any }[] = [
  { key: "standard", label: "Default", Icon: MapIcon },
  { key: "satellite", label: "Satellite", Icon: Satellite },
  { key: "terrain", label: "Terrain", Icon: Mountain },
];

const overlays: { key: FeatureKey; label: string; Icon: any }[] = [
  { key: "heat_signatures", label: "Heat Signatures", Icon: Flame },
  { key: "rainfall_events", label: "Rainfall Events", Icon: CloudRain },
  { key: "water_stress", label: "Water Stress", Icon: Droplets },
  { key: "growth_trends", label: "Growth Trends", Icon: TrendingUp },
  { key: "risk_heatmap", label: "Drought / Flood Risk Heatmap", Icon: AlertTriangle },
  { key: "yield_prediction", label: "Yield Prediction Contours", Icon: Sprout },
];

const MapLayersControl = ({ mapStyle, activeFeature, onStyleChange, onFeatureChange }: Props) => {
  const [open, setOpen] = useState(false);

  return (
   <div className="fixed  top-[25%] right-3 z-[999] flex flex-col gap-2">
      <div className="relative">
  <motion.button
    whileTap={{ scale: 0.9 }}
    onClick={() => setOpen(!open)}
    className="w-10 h-10 rounded-full bg-card shadow-lg border border-border/40 flex items-center justify-center relative"
  >
    <AnimatePresence mode="wait" initial={false}>
      {open ? (
        <motion.div
          key="close"
          initial={{ opacity: 0, rotate: -45 }}
          animate={{ opacity: 1, rotate: 0 }}
          exit={{ opacity: 0, rotate: 45 }}
          transition={{ duration: 0.15 }}
          className="absolute inset-0 flex items-center justify-center"
        >
          <X className="w-4 h-4 text-foreground" />
        </motion.div>
      ) : (
        <motion.div
          key="layers"
          initial={{ opacity: 0, rotate: 45 }}
          animate={{ opacity: 1, rotate: 0 }}
          exit={{ opacity: 0, rotate: -45 }}
          transition={{ duration: 0.15 }}
          className="absolute inset-0 flex items-center justify-center"
        >
          <Layers className="w-4 h-4 text-primary" />
        </motion.div>
      )}
    </AnimatePresence>
  </motion.button>

  <AnimatePresence>
    {open && (
      <motion.div
        initial={{ opacity: 0, scale: 0.9, x: 20 }}
        animate={{ opacity: 1, scale: 1, x: 0 }}
        exit={{ opacity: 0, scale: 0.9, x: 20 }}
        className="absolute top-0 right-14 bg-card rounded-2xl p-3 sm:p-4 w-[calc(100vw-5rem)] max-w-[340px] shadow-2xl border border-border/40 origin-top-right"
      >
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-border/40">
          <h3 className="font-bold text-foreground">Map Layers & Views</h3>
         
        </div>

        {/* SECTION 1: MAP TYPE */}
        <div className="mb-5">
          <p className="text-[11px] font-bold text-muted-foreground mb-3 uppercase tracking-widest">
            Map Type
          </p>
          <div className="grid grid-cols-3 gap-2">
            {styles.map(({ key, label, Icon }) => (
              <button
                key={key}
                onClick={() => onStyleChange(key)}
                className={`flex flex-col items-center justify-center gap-1.5 sm:gap-2 p-2 sm:p-3 rounded-xl border-2 transition-all ${
                  mapStyle === key
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border/40 bg-muted/30 text-muted-foreground hover:bg-muted/80 hover:border-border/80"
                }`}
              >
                <Icon className="w-4 h-4 sm:w-5 sm:h-5" />
                <span className="text-[9px] sm:text-[10px] font-bold text-center leading-tight truncate w-full">{label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* SECTION 2: MAP DETAILS / OVERLAYS */}
        <div>
          <p className="text-[11px] font-bold text-muted-foreground mb-3 uppercase tracking-widest">
            Map Details / Overlays
          </p>
          <div className="flex flex-col gap-1.5">
            {overlays.map(({ key, label, Icon }) => {
              const isActive = activeFeature === key;
              return (
                <button
                  key={key}
                  onClick={() => onFeatureChange(isActive ? "all" : key)}
                  className={`flex items-center justify-between px-2 sm:px-3 py-2 sm:py-2.5 rounded-xl border transition-all ${
                    isActive 
                      ? "bg-primary/5 border-primary/20" 
                      : "bg-transparent border-transparent hover:bg-muted/50"
                  }`}
                >
                  <div className="flex items-center gap-2 sm:gap-3 overflow-hidden">
                    <div className={`p-1.5 rounded-md shrink-0 ${isActive ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}`}>
                      <Icon className="w-3 h-3 sm:w-4 sm:h-4" />
                    </div>
                    <span className={`text-xs sm:text-sm font-medium truncate ${isActive ? "text-primary" : "text-foreground"}`}>
                      {label}
                    </span>
                  </div>
                  {/* Toggle Switch */}
                  <div className={`shrink-0 w-8 sm:w-9 h-4 sm:h-5 rounded-full p-0.5 transition-colors ${isActive ? "bg-primary" : "bg-muted-foreground/30"}`}>
                    <div className={`w-3 h-3 sm:w-4 sm:h-4 bg-white rounded-full shadow-sm transition-transform ${isActive ? "translate-x-4" : "translate-x-0"}`} />
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </motion.div>
    )}
  </AnimatePresence>
  </div>
      <div className="flex flex-col bg-card shadow-lg border border-border/40 rounded-full overflow-hidden">
        <button
          onClick={() => window.dispatchEvent(new CustomEvent("map:zoomIn"))}
          className="w-10 h-10 flex items-center justify-center hover:bg-muted/50 active:bg-muted transition-colors border-b border-border/40 text-foreground"
          aria-label="Zoom In"
        >
          <Plus className="w-5 h-5" />
        </button>
        <button
          onClick={() => window.dispatchEvent(new CustomEvent("map:zoomOut"))}
          className="w-10 h-10 flex items-center justify-center hover:bg-muted/50 active:bg-muted transition-colors text-foreground"
          aria-label="Zoom Out"
        >
          <Minus className="w-5 h-5" />
        </button>
      </div>

     
   </div>
  );
};

export default MapLayersControl;