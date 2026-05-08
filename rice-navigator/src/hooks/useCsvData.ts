import { useState, useEffect } from "react";
import { parseCSV, computeStateMetrics, type StateMetrics, type DailyRecord } from "@/data/csvProcessor";

interface UseCsvDataResult {
  metrics: Record<string, StateMetrics> | null;
  rawRecords: DailyRecord[];
  loading: boolean;
  error: string | null;
}

export function useCsvData(): UseCsvDataResult {
  const [metrics, setMetrics] = useState<Record<string, StateMetrics> | null>(null);
  const [rawRecords, setRawRecords] = useState<DailyRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    fetch("/national_processed_v2.csv")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        const records = parseCSV(text);
        const computed = computeStateMetrics(records);
        setRawRecords(records);
        setMetrics(computed);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return { metrics, rawRecords, loading, error };
}
