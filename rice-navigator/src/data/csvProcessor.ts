/**
 * CSV Data Processor
 * Parses national_processed_v2.csv and computes per-state geospatial metrics.
 * Columns: date, NDVI, EVI, NDWI, VV, VH, precipitation, temperature_2m, State, Year
 */

export interface DailyRecord {
  date: string;
  ndvi: number;
  evi: number;
  ndwi: number;
  vv: number;
  vh: number;
  precipitation: number;
  temperature: number;
  state: string;
  year: number;
}

export type PhenologyStage = "Vegetative" | "Tillering" | "Heading" | "Ripening" | "Dormant";
export type HealthStatus  = "Healthy" | "Stressed" | "Moderate";

export interface MonthlyPoint {
  date: string;       // "2024-05"
  ndvi: number;
  evi: number;
  precipitation: number;
  temperature: number;
}

export interface StateMetrics {
  [x: string]: any;
  state: string;
  // Latest snapshot
  latestNdvi: number;
  latestEvi: number;
  latestNdwi: number;
  // Trend (positive = growing, negative = declining)
  ndviTrend: number;
  ndviTrendPct: number; // percent change
  // Phenology
  phenologyStage: PhenologyStage;
  phenologyProgress: number; // 0–100
  // Climate
  avgTemperature: number;
  tempDeviation: number;     // vs previous year
  avgPrecipitation: number;
  precipDeviation: number;   // vs previous year
  hasTemperatureStress: boolean;
  maxStressStreak: number;   // consecutive days > 35°C during peak season
  // History
  ndviPrevYear: number;
  ndviDiff: number;          // Current vs Prev
  // Overall classification
  healthStatus: HealthStatus;
  // Time-series (monthly averages of latest year)
  timeSeries: MonthlyPoint[];
  // Historical comparison (monthly averages per year)
  byYear: Record<number, { months: string[]; ndvi: number[]; precipitation: number[] }>;
}

// ─── CSV Parser ───────────────────────────────────────────────────────────────

export function parseCSV(text: string): DailyRecord[] {
  const lines = text.trim().split(/\r?\n/);
  const records: DailyRecord[] = [];

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    if (cols.length < 10) continue;
    const ndvi = parseFloat(cols[1]);
    if (isNaN(ndvi)) continue;
    records.push({
      date:          cols[0].trim(),
      ndvi,
      evi:           parseFloat(cols[2]),
      ndwi:          parseFloat(cols[3]),
      vv:            parseFloat(cols[4]),
      vh:            parseFloat(cols[5]),
      precipitation: parseFloat(cols[6]),
      temperature:   parseFloat(cols[7]),
      state:         cols[8].trim(),
      year:          parseInt(cols[9]),
    });
  }
  return records;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function avg(arr: number[]): number {
  return arr.length === 0 ? 0 : arr.reduce((s, v) => s + v, 0) / arr.length;
}

function monthKey(date: string): string {
  return date.substring(0, 7); // "2024-05"
}

function monthNum(date: string): number {
  return parseInt(date.substring(5, 7));
}

function monthLabel(mm: string): string {
  const names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return names[parseInt(mm) - 1] ?? mm;
}

function derivePhenology(ndvi: number, trend: number, month: number): PhenologyStage {
  // Nigeria rice calendar: planting May-Jun, vegetative Jul-Aug, heading Sep-Oct, ripening Nov
  if (month <= 6 && ndvi < 0.30) return "Vegetative";
  if (trend > 0.005 && ndvi < 0.50) return "Tillering";
  if (ndvi >= 0.50 || (month >= 9 && month <= 10)) return "Heading";
  if (trend < -0.003 && ndvi >= 0.25) return "Ripening";
  return "Dormant";
}

function phenologyProgress(stage: PhenologyStage): number {
  return { Vegetative: 12, Tillering: 32, Heading: 62, Ripening: 85, Dormant: 0 }[stage];
}

// ─── Main computation ─────────────────────────────────────────────────────────

export function computeStateMetrics(records: DailyRecord[]): Record<string, StateMetrics> {
  // Group by state
  const grouped: Record<string, DailyRecord[]> = {};
  for (const r of records) {
    (grouped[r.state] ??= []).push(r);
  }

  const result: Record<string, StateMetrics> = {};

  for (const [state, recs] of Object.entries(grouped)) {
    // Sort chronologically
    const sorted = [...recs].sort((a, b) => a.date.localeCompare(b.date));

    // Use the most recent year available
    const latestYear = Math.max(...sorted.map(r => r.year));
    const latestRecs = sorted.filter(r => r.year === latestYear);

    const latest = latestRecs[latestRecs.length - 1];

    // ── NDVI weekly trend ──────────────────────────────────────────────────
    const last7   = latestRecs.slice(-7);
    const prev7   = latestRecs.slice(-14, -7);
    const ndviNow = avg(last7.map(r => r.ndvi));
    const ndviPrev = prev7.length ? avg(prev7.map(r => r.ndvi)) : ndviNow;
    const ndviTrend = ndviNow - ndviPrev;
    const ndviTrendPct = ndviPrev !== 0 ? (ndviTrend / ndviPrev) * 100 : 0;

    // ── Phenology ──────────────────────────────────────────────────────────
    const latestMonth = monthNum(latest.date);
    const stage = derivePhenology(latest.ndvi, ndviTrend, latestMonth);

    // ── Temperature stress: consecutive days > 35°C during peak (Aug–Oct) ─
    const peakRecs = latestRecs.filter(r => {
      const m = monthNum(r.date);
      return m >= 8 && m <= 10;
    });
    let maxStreak = 0, streak = 0;
    for (const r of peakRecs) {
      if (r.temperature > 35) { streak++; maxStreak = Math.max(maxStreak, streak); }
      else streak = 0;
    }

    // ── Climate averages ───────────────────────────────────────────────────
    const avgTemperature   = avg(latestRecs.map(r => r.temperature));
    const avgPrecipitation = avg(latestRecs.map(r => r.precipitation));

    // ── Historical Comparison (vs Previous Year) ───────────────────────────
    const prevYear = latestYear - 1;
    const prevRecs = sorted.filter(r => r.year === prevYear);
    
    let tempDeviation = 0;
    let precipDeviation = 0;
    let ndviPrevYear = ndviNow;
    let ndviDiff = 0;

    if (prevRecs.length > 0) {
      const avgTempPrev = avg(prevRecs.map(r => r.temperature));
      const avgPrecipPrev = avg(prevRecs.map(r => r.precipitation));
      ndviPrevYear = avg(prevRecs.map(r => r.ndvi));
      
      tempDeviation = avgTemperature - avgTempPrev;
      precipDeviation = avgPrecipitation - avgPrecipPrev;
      ndviDiff = ndviNow - ndviPrevYear;
    }

    // ── Health classification ──────────────────────────────────────────────
    const healthStatus: HealthStatus =
      latest.ndvi >= 0.45 && latest.ndwi > -0.40 ? "Healthy" :
      latest.ndvi < 0.30                          ? "Stressed" : "Moderate";

    // ── Monthly time-series for latest year ───────────────────────────────
    const monthlyMap: Record<string, { ndvi: number[]; evi: number[]; prec: number[]; temp: number[] }> = {};
    for (const r of latestRecs) {
      const k = monthKey(r.date);
      const m = (monthlyMap[k] ??= { ndvi: [], evi: [], prec: [], temp: [] });
      m.ndvi.push(r.ndvi); m.evi.push(r.evi);
      m.prec.push(r.precipitation); m.temp.push(r.temperature);
    }
    const timeSeries: MonthlyPoint[] = Object.entries(monthlyMap)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, m]) => ({
        date,
        ndvi: avg(m.ndvi), evi: avg(m.evi),
        precipitation: avg(m.prec), temperature: avg(m.temp),
      }));

    // ── Historical by-year monthly averages ───────────────────────────────
    const years = [...new Set(recs.map(r => r.year))].sort();
    const byYear: StateMetrics["byYear"] = {};
    for (const yr of years) {
      const yRecs = sorted.filter(r => r.year === yr);
      const ndviM: Record<string, number[]>  = {};
      const precM: Record<string, number[]>  = {};
      for (const r of yRecs) {
        const mm = r.date.substring(5, 7);
        (ndviM[mm] ??= []).push(r.ndvi);
        (precM[mm] ??= []).push(r.precipitation);
      }
      const months = Object.keys(ndviM).sort();
      byYear[yr] = {
        months: months.map(mm => monthLabel(mm)),
        ndvi:   months.map(mm => avg(ndviM[mm])),
        precipitation: months.map(mm => avg(precM[mm])),
      };
    }

    result[state] = {
      state,
      latestNdvi: latest.ndvi,
      latestEvi:  latest.evi,
      latestNdwi: latest.ndwi,
      ndviTrend,
      ndviTrendPct,
      phenologyStage: stage,
      phenologyProgress: phenologyProgress(stage),
      avgTemperature,
      tempDeviation,
      avgPrecipitation,
      precipDeviation,
      hasTemperatureStress: maxStreak >= 2,
      maxStressStreak: maxStreak,
      healthStatus,
      ndviPrevYear,
      ndviDiff,
      timeSeries,
      byYear,
    };
  }

  return result;
}

// ─── Feature filter logic ──────────────────────────────────────────────────────

export type FeatureKey =
  | "all"
  | "healthy_fields"
  | "ndvi_vigor"
  | "phenology"
  | "growth_trends"
  | "temp_stress"
  | "rainfall_patterns"
  | "historical"
  | "irrigation_junctions"
  | "risk_heatmap"
  | "yield_prediction"
  | "custom_zones"
  | "pest_hotspots"
  | "ensemble_breakdown"
  | "feature_importance"
  | "actionable_recommendations"
  | "heat_signatures"
  | "rainfall_events"
  | "water_stress";

/** Returns true if the state's metrics satisfy the selected feature filter */
export function statePassesFilter(
  state: string,
  metrics: Record<string, StateMetrics>,
  feature: FeatureKey
): boolean {
  const m = metrics[state];
  if (!m) return feature === "all";
  switch (feature) {
    case "all":                   return true;
    case "healthy_fields":        return m.healthStatus === "Healthy";
    case "ndvi_vigor":            return m.latestNdvi >= 0.30;
    case "phenology":             return m.phenologyStage !== "Dormant";
    case "growth_trends":         return m.ndviTrend > 0;
    case "temp_stress":           return m.hasTemperatureStress || m.avgTemperature >= 28.5;
    case "rainfall_patterns":     return true;
    case "historical":            return true;
    case "irrigation_junctions":  return true;
    case "risk_heatmap":          return true;
    case "yield_prediction":      return true;
    case "custom_zones":          return true;
    default:                      return true;
  }
}
