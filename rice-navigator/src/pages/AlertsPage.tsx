import { motion } from "framer-motion";
import {
  Bell,
  AlertTriangle,
  Bug,
  Droplets,
  ThermometerSun,
  MapPin,
  ExternalLink,
  Shield,
  Activity,
  TrendingUp,
  Zap,
  Globe,
  X,
} from "lucide-react";
import { riceFields, type RiceField } from "@/data/nigeriaRiceData";
import type { StateMetrics } from "@/data/csvProcessor";

interface Props {
  metrics: Record<string, StateMetrics> | null;
  onViewAlert: (field: RiceField) => void;
  onClose: () => void;
}

// ─── Data & Config ─────────────────────────────────────────────────────────────

const alertIcons: Record<string, React.ElementType> = {
  drought: ThermometerSun,
  pest: Bug,
  water: Droplets,
  heat: AlertTriangle,
  general: AlertTriangle,
};

const severityConfig = {
  critical: {
    badge: "bg-red-50 text-red-700 border border-red-100",
    label: "Critical",
    iconBg: "bg-red-50",
    iconColor: "text-red-500",
    accent: "bg-gradient-to-b from-red-400 to-red-600",
  },
  moderate: {
    badge: "bg-amber-50 text-amber-700 border border-amber-100",
    label: "Moderate",
    iconBg: "bg-amber-50",
    iconColor: "text-amber-500",
    accent: "bg-gradient-to-b from-amber-400 to-amber-500",
  },
} as const;

// ─── StatCard ──────────────────────────────────────────────────────────────────

const StatCard = ({
  icon: Icon,
  label,
  value,
  valueColor = "text-gray-900",
  iconBg,
  iconColor,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  valueColor?: string;
  iconBg: string;
  iconColor: string;
}) => (
  <div className="bg-white rounded-2xl p-4 border border-gray-100 shadow-sm">
    <div
      className={`w-8 h-8 rounded-xl ${iconBg} flex items-center justify-center mb-3`}
    >
      <Icon className={`w-4 h-4 ${iconColor}`} strokeWidth={1.8} />
    </div>
    <p className="text-[10px] uppercase tracking-widest font-medium text-gray-400 mb-1">
      {label}
    </p>
    <p className={`font-bold text-[1.7rem] leading-none tracking-tight ${valueColor}`}>
      {value}
    </p>
  </div>
);

// ─── Page ──────────────────────────────────────────────────────────────────────

const AlertsPage = ({ metrics, onViewAlert, onClose }: Props) => {
  // Dynamically generate alerts based on real metrics
  const allAlerts = riceFields
    .filter((f) => f.type === "stress")
    .map((f) => {
      const stateMetrics = metrics?.[f.state];
      let icon = "general";
      let title = `Stress Detected in ${f.state}`;
      let detail = f.details;
      let severity = "moderate";

      if (stateMetrics) {
        if (stateMetrics.avgTemperature > 30) {
          icon = "heat";
          title = `Heat Stress in ${f.state}`;
          detail = `Prolonged temperatures above 30°C may impact grain filling at ${f.name}.`;
          severity = stateMetrics.avgTemperature > 32 ? "critical" : "moderate";
        } else if (stateMetrics.avgPrecipitation < 4.5) {
          icon = "drought";
          title = `Drought Warning in ${f.state}`;
          detail = `Rainfall deficit detected. Monitor irrigation lines for ${f.name}.`;
          severity = stateMetrics.avgPrecipitation < 3.5 ? "critical" : "moderate";
        } else if (stateMetrics.latestNdvi < 0.25) {
          icon = "pest";
          title = `Vegetation Decline in ${f.state}`;
          detail = `Rapid NDVI drop at ${f.name}. Possible pest infestation or nutrient deficiency.`;
          severity = "critical";
        } else if (stateMetrics.latestNdwi < -0.4) {
          icon = "water";
          title = `Water Scarcity in ${f.state}`;
          detail = `Low NDWI indicates severe water stress at ${f.name}.`;
          severity = "critical";
        }
      }

      return {
        id: f.id,
        field: f,
        title,
        subtitle: `${f.state} • ${f.lga}`,
        detail,
        hectares: f.hectares,
        position: f.position,
        severity,
        icon,
      };
    });

  const criticalCount = allAlerts.filter((a) => a.severity === "critical").length;
  const moderateCount = allAlerts.filter((a) => a.severity === "moderate").length;
  const totalHectares = allAlerts.reduce((s, a) => s + a.hectares, 0);

  return (
    <motion.div
      className="h-full bg-[#f4f7f4] overflow-y-auto no-scrollbar pb-24 md:pb-6"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      <div>
        {/* ── Hero ── */}
        <div className="px-5 sm:px-8 pt-6 pb-6">
          {/* Header row */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span
                className="w-2 h-2 rounded-full bg-red-500 shrink-0"
                style={{ animation: "pulse 2s ease-in-out infinite" }}
              />
              <span className="text-[10px] uppercase tracking-[0.14em] text-red-500 font-bold">
                Live Feed
              </span>
            </div>
            <button
              onClick={onClose}
              className="md:hidden w-8 h-8 flex items-center justify-center bg-gray-200/50 rounded-full text-gray-600 hover:bg-gray-200"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-black leading-[1.1] tracking-tight mb-2">
            Regional Monitoring
          </h1>
          <p className="text-sm text-gray-500 max-w-sm leading-relaxed">
            Real-time intelligence and anomaly detection from the Nigerian agricultural corridor.
          </p>
        </div>

        {/* ── Stats ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 px-4 sm:px-8 py-2">
          <StatCard
            icon={Zap}
            label="Active Risks"
            value={allAlerts.length.toString()}
            iconBg="bg-emerald-50"
            iconColor="text-emerald-600"
          />
          <StatCard
            icon={AlertTriangle}
            label="Critical"
            value={criticalCount.toString()}
            valueColor="text-red-500"
            iconBg="bg-red-50"
            iconColor="text-red-500"
          />
          <StatCard
            icon={ThermometerSun}
            label="Moderate"
            value={moderateCount.toString()}
            valueColor="text-amber-500"
            iconBg="bg-amber-50"
            iconColor="text-amber-500"
          />
          <StatCard
            icon={TrendingUp}
            label="Hectares"
            value={totalHectares.toLocaleString()}
            iconBg="bg-emerald-50"
            iconColor="text-emerald-600"
          />
        </div>

        {/* ── Section label ── */}
        <p className="text-[10px] uppercase tracking-[0.14em] font-medium text-gray-400 px-4 sm:px-8 mt-6 mb-3">
          Active alerts
        </p>

        {/* ── Alert cards ── */}
        <div className="px-4 sm:px-8 grid grid-cols-1 xl:grid-cols-2 gap-3">
          {allAlerts.map((alert, index) => {
            const sev = severityConfig[alert.severity as keyof typeof severityConfig];
            const Icon = alertIcons[alert.icon] ?? AlertTriangle;

            return (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05, duration: 0.32, ease: "easeOut" }}
                className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden
                           hover:border-gray-200 hover:shadow-md transition-all duration-200"
              >
                <div className="flex">
                  {/* Left accent */}
                  <div className={`w-1 shrink-0 ${sev.accent}`} />

                  {/* Body */}
                  <div className="flex-1 p-4 min-w-0">
                    {/* Header */}
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <div
                          className={`w-7 h-7 rounded-full ${sev.iconBg} flex items-center justify-center shrink-0`}
                        >
                          <Icon className={`w-3.5 h-3.5 ${sev.iconColor}`} strokeWidth={2} />
                        </div>
                        <h3 className="font-semibold text-gray-800 text-sm leading-snug">
                          {alert.title}
                        </h3>
                      </div>
                      <span
                        className={`text-[9px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full shrink-0 mt-0.5 ${sev.badge}`}
                      >
                        {sev.label}
                      </span>
                    </div>

                    {/* Location */}
                    <div className="flex items-center gap-1 mb-2">
                      <MapPin className="w-2.5 h-2.5 text-gray-300 shrink-0" strokeWidth={2} />
                      <span className="text-[11px] text-gray-400">{alert.subtitle}</span>
                    </div>

                    {/* Detail */}
                    <p className="text-xs text-gray-500 leading-relaxed line-clamp-2 mb-3">
                      {alert.detail}
                    </p>

                    {/* Footer */}
                    <div className="flex items-center justify-between gap-2 flex-wrap mt-auto">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-100 px-2.5 py-1 rounded-full">
                          {alert.hectares.toLocaleString()} ha
                        </span>
                        <span className="text-[10px] text-gray-300 font-mono hidden sm:block">
                          {alert.position[0].toFixed(2)}°N {alert.position[1].toFixed(2)}°E
                        </span>
                      </div>
                      <button 
                        onClick={() => onViewAlert(alert.field)}
                        className="flex items-center gap-1.5 text-[11px] font-medium text-blue-700 bg-blue-50 border border-blue-100 px-4 py-1.5 rounded-full hover:bg-blue-100 transition-colors"
                      >
                        <ExternalLink className="w-3 h-3" strokeWidth={2} />
                        View on Map
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* ── Footer summary ── */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.42 }}
          className="mx-4 sm:mx-8 mt-6 mb-8 bg-white border border-gray-100 rounded-2xl overflow-hidden shadow-sm
                   flex flex-col sm:flex-row"
        >
          {/* Count block */}
          <div
            className="px-6 py-5 border-b border-gray-100 sm:border-b-0 sm:border-r
                     flex items-center gap-4 sm:flex-col sm:items-start sm:justify-center sm:min-w-[156px]"
          >
            <div className="w-11 h-11 rounded-xl bg-red-50 border border-red-100 flex items-center justify-center shrink-0">
              <Shield className="w-5 h-5 text-red-600" strokeWidth={1.8} />
            </div>
            <div>
              <p className="font-extrabold text-gray-900 text-[2rem] leading-none tracking-tight">
                {criticalCount}
              </p>
              <p className="text-[10px] uppercase tracking-wider text-gray-400 mt-1">
                Critical interventions
              </p>
            </div>
          </div>

          {/* Status */}
          <div className="flex-1 px-6 py-5 flex items-center gap-3">
            <span
              className="w-1.5 h-1.5 rounded-full bg-red-500 shrink-0"
              style={{ animation: "pulse 2s ease-in-out infinite" }}
            />
            <p className="text-sm text-gray-500 leading-relaxed">
              Sensor network active ·{" "}
              <span className="text-gray-800 font-semibold">
                {criticalCount > 0 ? "Authorize drone surveillance for critical stress sectors." : "All systems operational. No critical interventions required."}
              </span>
            </p>
          </div>
        </motion.div>

      </div>
    </motion.div>
  );
};

export default AlertsPage;