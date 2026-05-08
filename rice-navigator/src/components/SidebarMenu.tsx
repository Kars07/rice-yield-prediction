import { motion } from "framer-motion";
import {
  Leaf, TrendingUp, Activity, Thermometer, Droplets, History, AlertTriangle,
  Map, PenLine, Sprout, BarChart, Lightbulb, Box
} from "lucide-react";
import type { FeatureKey } from "@/data/csvProcessor";

interface Props {
  activeFeature: FeatureKey;
  onChange: (key: FeatureKey) => void;
}

const CATEGORIES = [
  {
    title: "CROP HEALTH",
    items: [
      { key: "healthy_fields", label: "Healthy Fields", icon: Leaf, color: "text-green-500" },
      { key: "ndvi_vigor", label: "NDVI / EVI Vigor", icon: TrendingUp, color: "text-yellow-500" },
      { key: "phenology", label: "Phenology Stages", icon: Sprout, color: "text-green-400" },
      { key: "growth_trends", label: "Growth Rate Trends", icon: Activity, color: "text-teal-400" },
    ]
  },
  {
    title: "CLIMATE & RISK",
    items: [
      { key: "temp_stress", label: "Temperature Stress", icon: Thermometer, color: "text-orange-400" },
      { key: "rainfall_patterns", label: "Rainfall Patterns", icon: Droplets, color: "text-blue-400" },
      { key: "historical", label: "Historical Comparison", icon: History, color: "text-blue-500" },
      { key: "pest_hotspots", label: "Pest Hotspots", icon: AlertTriangle, color: "text-red-400" },
    ]
  },
  {
    title: "INFRASTRUCTURE",
    items: [
      { key: "irrigation_junctions", label: "Irrigation Junctions", icon: Droplets, color: "text-blue-400" },
      { key: "custom_zones", label: "Custom Farm Zones", icon: PenLine, color: "text-purple-400" },
    ]
  },
  {
    title: "ANALYTICS",
    items: [
      { key: "ensemble_breakdown", label: "Ensemble Breakdown", icon: Box, color: "text-cyan-400" },
      { key: "feature_importance", label: "Feature Importance", icon: BarChart, color: "text-purple-400" },
      { key: "actionable_recommendations", label: "Actionable Recommendations", icon: Lightbulb, color: "text-yellow-400" },
    ]
  }
];

const SidebarMenu = ({ activeFeature, onChange }: Props) => {
  return (
    <motion.div
      className="fixed top-0 left-0 h-full w-[350px] bg-white/95 backdrop-blur-md shadow-xl z-[999] overflow-y-auto p-6"
      initial={{ x: -350 }}
      animate={{ x: 0 }}
      transition={{ type: "spring", damping: 25, stiffness: 200 }}
    >
      <div className="mb-8 mt-4">
        <h1 className="text-2xl font-bold text-gray-900 tracking-tight">
          {activeFeature.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
        </h1>
      </div>

      <div className="flex flex-col gap-6">
        {CATEGORIES.map((cat, i) => (
          <div key={i}>
            <h3 className="text-[11px] font-bold text-gray-400 tracking-widest uppercase mb-3">
              {cat.title}
            </h3>
            <div className="flex flex-wrap gap-2">
              {cat.items.map((item) => {
                const isActive = activeFeature === item.key;
                const Icon = item.icon;
                return (
                  <button
                    key={item.key}
                    onClick={() => onChange(item.key as FeatureKey)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-full text-xs font-semibold border transition-all ${
                      isActive
                        ? "bg-blue-500 text-white border-blue-500 shadow-md"
                        : "bg-gray-50 text-gray-700 border-gray-200 hover:border-blue-300 hover:bg-gray-100"
                    }`}
                  >
                    <Icon className={`w-3.5 h-3.5 ${isActive ? "text-white" : item.color}`} />
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

export default SidebarMenu;
