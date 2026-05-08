import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect, useMemo } from "react";
import {
  ChevronUp, ChevronDown, MapPin, X, Droplets, Sprout,
  ThermometerSun, CloudRain, AlertTriangle, Database, Activity, TrendingUp, TrendingDown, History
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, AreaChart
} from "recharts";
import type { RiceField } from "@/data/nigeriaRiceData";
import type { StateMetrics } from "@/data/csvProcessor";

interface Props {
  field: RiceField | null;
  csvMetrics: Record<string, StateMetrics> | null;
  onClose: () => void;
}

const statusBadge: Record<RiceField["type"], { label: string; bg: string; text: string }> = {
  healthy:    { label: "HEALTHY",    bg: "bg-[hsl(var(--field-healthy))]/15",    text: "text-[hsl(var(--field-healthy))]" },
  stress:     { label: "STRESS",     bg: "bg-destructive/15",                    text: "text-destructive" },
  irrigation: { label: "IRRIGATION", bg: "bg-[hsl(var(--field-irrigation))]/15", text: "text-[hsl(var(--field-irrigation))]" },
  growth:     { label: "GROWING",    bg: "bg-[hsl(var(--field-growth))]/15",     text: "text-[hsl(var(--field-growth))]" },
};

// Map NDWI (-0.6..0) to 10..90%
const soilFromNdwi = (ndwi: number): number => {
  const clamped = Math.max(-0.6, Math.min(0, ndwi));
  return Math.round(10 + ((clamped + 0.6) / 0.6) * 80);
};

const yieldFromNdvi = (ndvi: number): string => {
  const tpha = Math.max(0.5, Math.min(6.5, 1 + ndvi * 7));
  return `${tpha.toFixed(1)} t/ha`;
};

// Stable pseudo-random multiplier (0.92 to 1.08) based on field ID
const getFarmVariance = (id: string) => {
  const hash = id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return 0.92 + (hash % 17) / 100;
};

const InfoSheet = ({ field, csvMetrics, onClose }: Props) => {
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setExpanded(false);
  }, [field?.id]);

  const metrics = field ? csvMetrics?.[field.state] : undefined;
  const hasRealData = !!metrics;

  const variance = field ? getFarmVariance(field.id) : 1;

  const soilPct  = metrics ? Math.min(100, Math.round(soilFromNdwi(metrics.latestNdwi) * variance)) : null;
  const yieldStr = metrics ? yieldFromNdvi(metrics.latestNdvi * variance) : null;
  const stage    = metrics ? metrics.phenologyStage : null;
  const progress = metrics ? Math.min(100, Math.round(metrics.phenologyProgress * variance)) : 0;
  const displayTemp = metrics ? metrics.avgTemperature * (1 + (variance - 1) * 0.2) : null; // Temp varies less
  
  // Format data for the Historical Rainfall Chart
  const historicalData = useMemo(() => {
    if (!metrics?.byYear) return [];
    const yrs = Object.keys(metrics.byYear).map(Number).sort();
    if (yrs.length === 0) return [];
    
    // Base it on the months of the latest year
    const latestYr = yrs[yrs.length - 1];
    const months = metrics.byYear[latestYr].months;
    
    return months.map((month, idx) => {
      const dataPoint: any = { month };
      yrs.forEach(yr => {
        dataPoint[`yr${yr}`] = metrics.byYear[yr].precipitation[idx] ?? 0;
      });
      return dataPoint;
    });
  }, [metrics]);

  return (
    <AnimatePresence>
      {field && (
        <motion.div
          className="fixed md:bottom-0 bottom-20 left-3 md:left-24 right-3 z-[1000] flex justify-center pointer-events-none"
          initial={{ y: 240, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 240, opacity: 0 }}
          transition={{ type: "spring", damping: 24, stiffness: 280 }}
        >
          <div className="bg-card w-full max-w-2xl rounded-2xl shadow-2xl border border-border/40 overflow-hidden pointer-events-auto flex flex-col max-h-[85vh]">
            
            {/* Drag handle */}
            <button
              onClick={() => setExpanded(!expanded)}
              className="w-full flex justify-center pt-2 pb-1 cursor-pointer shrink-0"
            >
              <div className="w-10 h-1 rounded-full bg-muted-foreground/30" />
            </button>

            <div className="px-5 pb-5 overflow-y-auto no-scrollbar">
              {/* Header */}
              <div className="flex items-start justify-between gap-2 mb-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider shrink-0 ${statusBadge[field.type].bg} ${statusBadge[field.type].text}`}>
                      {statusBadge[field.type].label}
                    </span>
                    <span className="text-[11px] text-muted-foreground truncate">{field.state} • {field.lga}</span>
                    {hasRealData ? (
                      <span className="text-[9px] px-1.5 py-0.5 rounded-full font-bold bg-emerald-500/15 text-emerald-600 shrink-0">
                        CSV DATA
                      </span>
                    ) : (
                      <span className="text-[9px] px-1.5 py-0.5 rounded-full font-bold bg-muted text-muted-foreground shrink-0">
                        NO DATA
                      </span>
                    )}
                  </div>
                  <h3 className="text-xl font-bold text-foreground truncate leading-tight">{field.name}</h3>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    onClick={() => setExpanded(!expanded)}
                    className="w-8 h-8 rounded-full bg-muted flex items-center justify-center hover:bg-muted/80 transition-colors"
                    aria-label={expanded ? "Collapse" : "Expand"}
                  >
                    {expanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronUp className="w-4 h-4 text-muted-foreground" />}
                  </button>
                  <button onClick={onClose} className="w-8 h-8 rounded-full bg-muted flex items-center justify-center hover:bg-muted/80 transition-colors" aria-label="Close">
                    <X className="w-4 h-4 text-muted-foreground" />
                  </button>
                </div>
              </div>

              {/* Phenology Progress Bar */}
              {hasRealData && (
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs text-muted-foreground font-semibold uppercase tracking-wider flex items-center gap-1.5">
                      <Sprout className="w-3.5 h-3.5" /> Phenology Stage
                    </span>
                    <span className="text-sm font-bold text-primary">{stage}</span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden relative">
                    <motion.div
                      className="absolute top-0 left-0 bottom-0 bg-primary rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                    />
                    {/* Stage markers */}
                    <div className="absolute inset-0 flex justify-between px-1">
                      {[25, 50, 75].map(p => (
                        <div key={p} className="h-full w-px bg-background/40" style={{ left: `${p}%`, position: 'absolute' }} />
                      ))}
                    </div>
                  </div>
                  <div className="flex justify-between text-[9px] text-muted-foreground font-medium mt-1 uppercase">
                    <span>Veg</span>
                    <span>Tillering</span>
                    <span>Heading</span>
                    <span>Ripening</span>
                  </div>
                </div>
              )}

              {/* Quick stats grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-2">
                <div className="bg-muted/40 rounded-xl p-3 border border-border/40">
                  <p className="text-[10px] text-muted-foreground uppercase font-semibold mb-1 flex items-center gap-1">
                    <Droplets className="w-3 h-3" /> Soil Moisture
                  </p>
                  <p className="text-lg font-bold text-foreground">{soilPct !== null ? `${soilPct}%` : "--"}</p>
                </div>
                <div className="bg-muted/40 rounded-xl p-3 border border-border/40">
                  <p className="text-[10px] text-muted-foreground uppercase font-semibold mb-1 flex items-center gap-1">
                    <Activity className="w-3 h-3" /> Est. Yield
                  </p>
                  <p className="text-lg font-bold text-foreground">{yieldStr ?? "--"}</p>
                </div>
                <div className="bg-muted/40 rounded-xl p-3 border border-border/40">
                  <p className="text-[10px] text-muted-foreground uppercase font-semibold mb-1 flex items-center gap-1">
                    <ThermometerSun className="w-3 h-3" /> Avg Temp
                  </p>
                  <p className="text-lg font-bold text-foreground">
                    {displayTemp ? `${displayTemp.toFixed(1)}°` : "--"}
                    {metrics?.hasTemperatureStress && <AlertTriangle className="w-3 h-3 text-destructive inline ml-1 mb-1" />}
                  </p>
                </div>
                <div className="bg-muted/40 rounded-xl p-3 border border-border/40">
                  <p className="text-[10px] text-muted-foreground uppercase font-semibold mb-1 flex items-center gap-1">
                    <MapPin className="w-3 h-3" /> Area
                  </p>
                  <p className="text-lg font-bold text-foreground">{field.hectares} <span className="text-xs text-muted-foreground font-normal">ha</span></p>
                </div>
              </div>

              {/* Expanded Data Visualizations */}
              <AnimatePresence initial={false}>
                {expanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                  >
                    <div className="pt-4 mt-2 border-t border-border/40">
                      
                      {hasRealData ? (
                        <div className="space-y-6">
                          
                          {/* 1. Growth Rate Trend (NDVI Vigor) */}
                          <div className="bg-card border border-border/50 rounded-xl p-4 shadow-sm">
                            <div className="flex items-start justify-between mb-4">
                              <div>
                                <h4 className="text-sm font-bold text-foreground flex items-center gap-1.5">
                                  <TrendingUp className="w-4 h-4 text-primary" /> NDVI / EVI Vigor
                                </h4>
                                <p className="text-[11px] text-muted-foreground">Current season vegetation indices</p>
                              </div>
                              <div className={`flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold ${metrics.ndviTrend > 0 ? "bg-emerald-500/15 text-emerald-600" : "bg-destructive/15 text-destructive"}`}>
                                {metrics.ndviTrend > 0 ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                                {Math.abs(metrics.ndviTrendPct).toFixed(1)}% / wk
                              </div>
                            </div>
                            
                            <div className="h-[160px] w-full mt-2">
                              <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={metrics.timeSeries} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                                  <defs>
                                    <linearGradient id="colorNdvi" x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/>
                                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                                    </linearGradient>
                                  </defs>
                                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} tickFormatter={(val) => val.substring(5,7)} axisLine={false} tickLine={false} />
                                  <YAxis tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} domain={[0, 'auto']} />
                                  <Tooltip 
                                    contentStyle={{ backgroundColor: 'hsl(var(--card))', borderRadius: '8px', border: '1px solid hsl(var(--border))', fontSize: '12px' }}
                                    itemStyle={{ fontWeight: 'bold' }}
                                  />
                                  <Area type="monotone" dataKey="ndvi" name="NDVI" stroke="#22c55e" strokeWidth={2} fillOpacity={1} fill="url(#colorNdvi)" />
                                  <Line type="monotone" dataKey="evi" name="EVI" stroke="#eab308" strokeWidth={2} dot={false} />
                                  <Legend wrapperStyle={{ fontSize: '10px' }} />
                                </AreaChart>
                              </ResponsiveContainer>
                            </div>
                          </div>

                          {/* 2. Historical Comparison (Rainfall) */}
                          <div className="bg-card border border-border/50 rounded-xl p-4 shadow-sm">
                            <div className="mb-4">
                              <h4 className="text-sm font-bold text-foreground flex items-center gap-1.5">
                                <History className="w-4 h-4 text-blue-500" /> Historical Rainfall Comparison
                              </h4>
                              <p className="text-[11px] text-muted-foreground">Precipitation (mm) across years</p>
                            </div>
                            
                            <div className="h-[160px] w-full">
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={historicalData} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                                  <XAxis dataKey="month" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} />
                                  <YAxis tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} axisLine={false} tickLine={false} />
                                  <Tooltip 
                                    contentStyle={{ backgroundColor: 'hsl(var(--card))', borderRadius: '8px', border: '1px solid hsl(var(--border))', fontSize: '12px' }}
                                  />
                                  {Object.keys(metrics.byYear).sort().map((yr, idx, arr) => {
                                    const isLatest = idx === arr.length - 1;
                                    return (
                                      <Line 
                                        key={yr}
                                        type="monotone" 
                                        dataKey={`yr${yr}`} 
                                        name={`${yr}`} 
                                        stroke={isLatest ? "#3b82f6" : "hsl(var(--muted-foreground))"} 
                                        strokeWidth={isLatest ? 3 : 1.5}
                                        strokeDasharray={isLatest ? "0" : "4 4"}
                                        dot={isLatest ? { r: 3, fill: '#3b82f6' } : false} 
                                      />
                                    );
                                  })}
                                  <Legend wrapperStyle={{ fontSize: '10px' }} />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          </div>

                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center py-8 px-4 text-center bg-muted/20 rounded-xl border border-dashed border-border">
                          <Database className="w-8 h-8 text-muted-foreground/50 mb-3" />
                          <p className="text-sm font-semibold text-foreground mb-1">No historical data found</p>
                          <p className="text-xs text-muted-foreground max-w-xs leading-relaxed">
                            Detailed charts and metrics are only available for farms in Ebonyi, Jigawa, Kano, Kebbi, Niger & Taraba.
                          </p>
                        </div>
                      )}
                      
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default InfoSheet;