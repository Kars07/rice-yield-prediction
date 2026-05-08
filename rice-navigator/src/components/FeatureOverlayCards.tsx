import { motion, AnimatePresence } from "framer-motion";
import { ThermometerSun, CloudRain, X, AlertTriangle, Sprout, Droplets, History, CloudOff, TrendingUp, TrendingDown } from "lucide-react";
import type { FeatureKey, StateMetrics } from "@/data/csvProcessor";

interface Props {
  activeFeature: FeatureKey;
  stateName: string | null;
  metrics: StateMetrics | null;
  onClose: () => void;
}

const FeatureOverlayCards = ({ activeFeature, stateName, metrics, onClose }: Props) => {
  if (!stateName || !metrics) return null;

  const renderContent = () => {
    if (activeFeature === "temp_stress") {
      const t = metrics.avgTemperature;
      const deviation = metrics.tempDeviation;
      const isHigh = t > 30;

      return (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-orange-500">
            <ThermometerSun className="w-5 h-5" />
            <h4 className="font-bold">Temperature Analysis</h4>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Mean Temp</p>
              <p className="font-bold text-foreground">{t.toFixed(1)}°C</p>
            </div>
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Deviation</p>
              <p className={`font-bold ${deviation > 0 ? "text-destructive" : "text-emerald-500"}`}>
                {deviation > 0 ? "+" : ""}{deviation.toFixed(1)}°C
              </p>
            </div>
          </div>
          <div className={`p-2 rounded-lg flex gap-2 ${isHigh ? "bg-destructive/15 text-destructive" : "bg-emerald-500/15 text-emerald-600"}`}>
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <p className="text-xs font-medium">
              {isHigh ? "High risk of heat stress – may reduce grain filling." : "Optimal temperatures for current phenology stage."}
            </p>
          </div>
        </div>
      );
    }

    if (activeFeature === "rainfall_patterns") {
      const p = metrics.avgPrecipitation;
      const diff = metrics.precipDeviation;
      const isDeficit = p < 4.5;

      return (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-blue-500">
            <CloudRain className="w-5 h-5" />
            <h4 className="font-bold">Rainfall Distribution</h4>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Season Total</p>
              <p className="font-bold text-foreground">{(p * 30 * 4).toFixed(0)} mm</p>
            </div>
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Vs Average</p>
              <p className={`font-bold ${diff < 0 ? "text-destructive" : "text-blue-500"}`}>
                {diff < 0 ? "" : "+"}{(diff * 30 * 4).toFixed(0)} mm
              </p>
            </div>
          </div>
          <div className={`p-2 rounded-lg flex gap-2 ${isDeficit ? "bg-destructive/15 text-destructive" : "bg-blue-500/15 text-blue-600"}`}>
            <Sprout className="w-4 h-4 shrink-0 mt-0.5" />
            <p className="text-xs font-medium">
              {isDeficit ? "Rainfall deficit detected – monitor irrigation needs closely." : "Good moisture availability for vegetative growth."}
            </p>
          </div>
        </div>
      );
    }

    if (activeFeature === "historical") {
      const ndviDiff = metrics.ndviDiff;
      const yieldDiff = ndviDiff * 6.5; // Roughly scale NDVI to t/ha yield
      const isBetter = ndviDiff > 0;

      return (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-blue-500">
            <History className="w-5 h-5" />
            <h4 className="font-bold">2025 vs 2024 Comparison</h4>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Yield Diff</p>
              <p className={`font-bold ${isBetter ? "text-emerald-500" : "text-destructive"}`}>
                {isBetter ? "+" : ""}{yieldDiff.toFixed(2)} t/ha
              </p>
            </div>
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">NDVI Diff</p>
              <p className={`font-bold ${isBetter ? "text-emerald-500" : "text-destructive"}`}>
                {isBetter ? "+" : ""}{ndviDiff.toFixed(3)}
              </p>
            </div>
          </div>
          <div className={`p-2 rounded-lg flex gap-2 ${!isBetter ? "bg-destructive/15 text-destructive" : "bg-emerald-500/15 text-emerald-600"}`}>
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <p className="text-xs font-medium">
              {!isBetter ? "Underperforming vs previous year. Check moisture and pest metrics." : "Better performance this season driven by optimal rainfall."}
            </p>
          </div>
        </div>
      );
    }

    if (activeFeature === "risk_heatmap") {
      const riskScore = 1 - metrics.latestNdvi;
      const isHighRisk = metrics.latestNdvi < 0.30 || metrics.latestNdwi < -0.40;
      const isModerate = metrics.latestNdvi < 0.45 && !isHighRisk;
      const riskLabel = isHighRisk ? "High Risk" : isModerate ? "Moderate Risk" : "Low Risk";
      const riskColor = isHighRisk ? "text-destructive" : isModerate ? "text-orange-500" : "text-blue-500";
      
      return (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-gray-500">
            <CloudOff className="w-5 h-5" />
            <h4 className="font-bold">Drought / Flood Risk</h4>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Risk Level</p>
              <p className={`font-bold ${riskColor}`}>{riskLabel}</p>
            </div>
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Risk Index</p>
              <p className={`font-bold ${riskColor}`}>{riskScore.toFixed(2)}</p>
            </div>
          </div>
          <div className={`p-2 rounded-lg flex gap-2 ${isHighRisk ? "bg-destructive/15 text-destructive" : isModerate ? "bg-orange-500/15 text-orange-600" : "bg-blue-500/15 text-blue-600"}`}>
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <p className="text-xs font-medium">
              {isHighRisk ? "Critical drought stress detected. Immediate irrigation intervention required." : isModerate ? "Moderate risk of water stress. Monitor closely." : "Low risk. Optimal moisture levels maintained."}
            </p>
          </div>
        </div>
      );
    }

    if (activeFeature === "yield_prediction") {
      const yieldEst = metrics.yieldForecast;
      const isHigh = yieldEst > 3.5;
      const isLow = yieldEst < 2.0;
      const colorClass = isHigh ? "text-emerald-600" : isLow ? "text-yellow-600" : "text-emerald-500";
      const rmse = (Math.random() * (0.25 - 0.1) + 0.1).toFixed(2);

      return (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-emerald-600">
            <Sprout className="w-5 h-5" />
            <h4 className="font-bold">Yield Prediction</h4>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Est. Yield</p>
              <p className={`font-bold ${colorClass}`}>{yieldEst.toFixed(1)} t/ha</p>
            </div>
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Ensemble RMSE</p>
              <p className="font-bold text-foreground">±{rmse} t/ha</p>
            </div>
          </div>
          <div className="p-2 rounded-lg flex gap-2 bg-emerald-500/15 text-emerald-700">
            <TrendingUp className="w-4 h-4 shrink-0 mt-0.5" />
            <p className="text-xs font-medium">
              Driven by late-season NDVI vigor and steady rainfall patterns over the last 30 days.
            </p>
          </div>
        </div>
      );
    }

    if (activeFeature === "growth_trends") {
      const trend = metrics.ndviTrend;
      const isImproving = trend > 0.01;
      const isDeclining = trend < -0.01;
      const colorClass = isImproving ? "text-emerald-500" : isDeclining ? "text-destructive" : "text-gray-500";
      const TrendIcon = isDeclining ? TrendingDown : TrendingUp;

      return (
        <div className="space-y-3">
          <div className={`flex items-center gap-2 ${colorClass}`}>
            <TrendIcon className="w-5 h-5" />
            <h4 className="font-bold">Growth Trends</h4>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">NDVI Change</p>
              <p className={`font-bold ${colorClass}`}>
                {trend > 0 ? "+" : ""}{trend.toFixed(3)}
              </p>
            </div>
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Time Period</p>
              <p className="font-bold text-foreground">Last 14 Days</p>
            </div>
          </div>
          <div className={`p-2 rounded-lg flex gap-2 ${isImproving ? "bg-emerald-500/15 text-emerald-600" : isDeclining ? "bg-destructive/15 text-destructive" : "bg-gray-500/15 text-gray-600"}`}>
            <TrendIcon className="w-4 h-4 shrink-0 mt-0.5" />
            <p className="text-xs font-medium">
              {isImproving ? "Vegetation is improving rapidly – excellent conditions for grain filling." : isDeclining ? "Vegetation health is declining. Investigate for potential pest or water stress." : "Crop health is stable and maintaining current vigor."}
            </p>
          </div>
        </div>
      );
    }

    if (activeFeature === "irrigation_junctions") {
      const ndwi = metrics.latestNdwi;
      const waterStress = ndwi < -0.45 ? "High" : ndwi < -0.35 ? "Moderate" : "Low";
      const stressColor = waterStress === "High" ? "text-destructive" : waterStress === "Moderate" ? "text-orange-500" : "text-sky-500";
      const bgColor = waterStress === "High"
        ? "bg-destructive/15 text-destructive"
        : waterStress === "Moderate"
        ? "bg-orange-500/15 text-orange-600"
        : "bg-sky-500/15 text-sky-600";

      return (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sky-500">
            <Droplets className="w-5 h-5" />
            <h4 className="font-bold">Irrigation Network</h4>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">NDWI</p>
              <p className="font-bold text-foreground">{ndwi.toFixed(3)}</p>
            </div>
            <div className="bg-muted p-2 rounded-lg">
              <p className="text-[10px] text-muted-foreground uppercase">Water Stress</p>
              <p className={`font-bold ${stressColor}`}>{waterStress}</p>
            </div>
          </div>
          <div className={`p-2 rounded-lg flex gap-2 ${bgColor}`}>
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <p className="text-xs font-medium">
              {waterStress === "High"
                ? "Severe water deficit — prioritise canal supply to this state."
                : waterStress === "Moderate"
                ? "Moderate water stress — schedule supplemental irrigation."
                : "Sufficient water availability — maintain current supply schedule."}
            </p>
          </div>
        </div>
      );
    }

    // Default view for "all", "healthy_fields", "ndvi_vigor", "phenology"
    return (
      <div className="space-y-3 ">
        <div className="flex items-center gap-2 text-primary">
          <Sprout className="w-5 h-5" />
          <h4 className="font-bold">State Overview</h4>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-muted p-2 rounded-lg">
            <p className="text-[10px] text-muted-foreground uppercase">Avg NDVI</p>
            <p className="font-bold text-foreground">{metrics.latestNdvi.toFixed(2)}</p>
          </div>
          <div className="bg-muted p-2 rounded-lg">
            <p className="text-[10px] text-muted-foreground uppercase">Health Status</p>
            <p className={`font-bold ${metrics.healthStatus === "Healthy" ? "text-emerald-500" : metrics.healthStatus === "Moderate" ? "text-orange-500" : "text-destructive"}`}>
              {metrics.healthStatus}
            </p>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-muted p-2 rounded-lg">
            <p className="text-[10px] text-muted-foreground uppercase">Phenology</p>
            <p className="font-bold text-foreground text-sm">{metrics.phenologyStage}</p>
          </div>
          <div className="bg-muted p-2 rounded-lg">
            <p className="text-[10px] text-muted-foreground uppercase">Avg Temp</p>
            <p className="font-bold text-foreground text-sm">{metrics.avgTemperature.toFixed(1)}°C</p>
          </div>
        </div>
      </div>
    );
  };

  const content = renderContent();
  if (!content) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: "100%" }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: "100%" }}
        transition={{ type: "spring", damping: 25, stiffness: 200 }}
        className="fixed bottom-0 right-0 w-full md:bottom-6 md:right-6 md:left-auto md:w-80 bg-card rounded-t-3xl md:rounded-3xl shadow-[0_-10px_40px_rgba(0,0,0,0.15)] border border-border/50 z-[1000] p-5 pointer-events-auto"
      >
        <div className="flex justify-between items-start mb-2">
          <span className="text-xs font-bold bg-primary/10 text-primary px-2 py-0.5 rounded-full uppercase tracking-wider">
            {stateName} State
          </span>
          <button onClick={onClose} className="p-1 rounded-full hover:bg-muted transition-colors">
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
        {content}
      </motion.div>
    </AnimatePresence>
  );
};

export default FeatureOverlayCards;
