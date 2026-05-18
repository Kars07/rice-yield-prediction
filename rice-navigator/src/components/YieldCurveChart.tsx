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
            // Use a dark hex — lightweight-charts does NOT support 'transparent'
            background: { type: ColorType.Solid, color: "#0f1117" },
            textColor: "#9ca3af",
            fontSize: 11,
          },
          grid: {
            vertLines: { color: "rgba(255,255,255,0.04)" },
            horzLines: { color: "rgba(255,255,255,0.06)" },
          },
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight || 400,
          rightPriceScale: {
            borderColor: "rgba(255,255,255,0.08)",
            textColor: "#6b7280",
            scaleMargins: { top: 0.15, bottom: 0.15 },
          },
          timeScale: {
            borderColor: "rgba(255,255,255,0.08)",
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

        // State Area Curve — emerald fill
        const stateArea = chart.addSeries(AreaSeries, {
          lineColor: "#10b981",
          topColor: "rgba(16,185,129,0.35)",
          bottomColor: "rgba(16,185,129,0.02)",
          lineWidth: 2,
          crosshairMarkerRadius: 5,
          crosshairMarkerBorderColor: "#ffffff",
          priceLineVisible: true,
          lastValueVisible: true,
        });
        stateArea.setData(data.state_yield_curve);

        // Regional Dashed Line — violet
        const regionLine = chart.addSeries(LineSeries, {
          color: "#8b5cf6",
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          crosshairMarkerRadius: 4,
          crosshairMarkerBorderColor: "#ffffff",
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
      <div className="rounded-xl overflow-hidden border border-white/8 relative grow min-h-0" style={{ background: "#0f1117", height }}>
        {loading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-[#0f1117] text-emerald-500/60">
            <div className="w-5 h-5 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
            <span className="text-xs font-medium animate-pulse">Running LSTM + XGBoost ensemble...</span>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-[#0f1117] text-red-400 text-center px-6">
            <span className="text-xl">⚠</span>
            <p className="text-xs font-medium">{error}</p>
            <p className="text-[10px] text-muted-foreground">The cloud model API might be spinning up or temporarily unreachable.</p>
          </div>
        )}
        
        <div ref={chartContainerRef} className="w-full h-full" />
      </div>
    </div>
  );
}
