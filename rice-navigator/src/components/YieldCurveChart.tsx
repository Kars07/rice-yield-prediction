import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, LineStyle, AreaSeries, LineSeries } from "lightweight-charts";

interface Props {
  stateName: string;
  height?: number | string;
}

export function YieldCurveChart({ stateName, height = "100%" }: Props) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [predictedYield, setPredictedYield] = useState<number | null>(null);

  useEffect(() => {
    let chart: ReturnType<typeof createChart> | null = null;
    let resizeHandler: (() => void) | null = null;

    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        setPredictedYield(null);

        const res = await fetch("https://rice-yield-prediction.onrender.com/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ state: stateName }),
        });

        if (!res.ok) throw new Error(`Server error: ${res.status} ${res.statusText}`);

        const data = await res.json();
        if (data.error) throw new Error(data.error);
        if (!data.state_yield_curve?.length) throw new Error("No curve data returned from API");

        setPredictedYield(data.predicted_yield);
        setLoading(false); // Set loading false so the container is rendered

        // Small delay to ensure DOM is painted before chart mounts
        await new Promise((r) => setTimeout(r, 50));

        if (!chartContainerRef.current) return;

        chart = createChart(chartContainerRef.current, {
          layout: {
            background: { type: ColorType.Solid, color: "#ffffff" },
            textColor: "#334155",
            fontSize: 11,
          },
          grid: {
            vertLines: { color: "rgba(0,0,0,0.05)" },
            horzLines: { color: "rgba(0,0,0,0.05)" },
          },
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight || 400,
          rightPriceScale: {
            borderColor: "rgba(0,0,0,0.08)",
            textColor: "#475569",
            scaleMargins: { top: 0.15, bottom: 0.15 },
          },
          timeScale: {
            borderColor: "rgba(0,0,0,0.08)",
            fixLeftEdge: true,
            fixRightEdge: true,
            timeVisible: true,
            secondsVisible: false,
            tickMarkFormatter: (time: any) => String(time).slice(0, 4), // show only year
          },
          crosshair: { mode: 1 },
          handleScroll: true,
          handleScale: true,
        });

        // State Area Curve — deeper, highly vibrant emerald fill for perfect popping
        const stateArea = chart.addSeries(AreaSeries, {
          lineColor: "#059669",
          topColor: "rgba(5,150,105,0.22)",
          bottomColor: "rgba(5,150,105,0.01)",
          lineWidth: 3.5,
          crosshairMarkerRadius: 6,
          crosshairMarkerBorderColor: "#059669",
          priceLineVisible: true,
          lastValueVisible: true,
        });
        stateArea.setData(data.state_yield_curve);

        // Regional Dashed Line — deeper violet for perfect contrast on white
        const regionLine = chart.addSeries(LineSeries, {
          color: "#6d28d9",
          lineWidth: 2.5,
          lineStyle: LineStyle.Dashed,
          crosshairMarkerRadius: 5,
          crosshairMarkerBorderColor: "#6d28d9",
          priceLineVisible: false,
          lastValueVisible: true,
        });
        regionLine.setData(data.region_yield_curve);

        chart.timeScale().fitContent();

        resizeHandler = () => {
          if (chartContainerRef.current && chart) {
            chart.applyOptions({ 
              width: chartContainerRef.current.clientWidth,
              height: chartContainerRef.current.clientHeight 
            });
          }
        };
        window.addEventListener("resize", resizeHandler);
      } catch (e: any) {
        setError(e.message ?? "Unknown error");
        setLoading(false);
      }
    };

    loadData();

    return () => {
      if (resizeHandler) window.removeEventListener("resize", resizeHandler);
      if (chart) chart.remove();
    };
  }, [stateName, height]);

  return (
    <div className="w-full h-full flex flex-col gap-2 min-h-0">
      {/* Legend row */}
      <div className="flex items-center justify-between px-1">
        <div className="flex gap-4">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 bg-emerald-500 rounded-full" />
            <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">{stateName}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 bg-violet-500 rounded-full opacity-70 border-dashed" style={{ borderTop: "2px dashed #8b5cf6", background: "none" }} />
            <span className="text-[10px] text-violet-400 font-bold uppercase tracking-wider">Regional Avg</span>
          </div>
        </div>
        {predictedYield !== null && (
          <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full">
            Latest: {predictedYield.toFixed(2)} t/ha
          </span>
        )}
      </div>

      {/* Chart area */}
      <div className="rounded-xl overflow-hidden border border-slate-200 relative grow min-h-0" style={{ background: "#ffffff", height }}>
        {loading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-white text-emerald-600">
            <div className="w-5 h-5 border-2 border-emerald-500/30 border-t-emerald-600 rounded-full animate-spin" />
            <span className="text-xs font-semibold animate-pulse text-slate-700">Running LSTM + XGBoost ensemble...</span>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-white text-red-500 text-center px-6">
            <span className="text-xl">⚠</span>
            <p className="text-xs font-semibold text-slate-800">{error}</p>
            <p className="text-[10px] text-slate-500">The cloud model API might be spinning up or temporarily unreachable.</p>
          </div>
        )}
        
        <div ref={chartContainerRef} className="w-full h-full" />
      </div>
    </div>
  );
}
