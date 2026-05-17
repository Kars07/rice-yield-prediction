import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, MapPin, TrendingUp, Cpu, Leaf } from "lucide-react";
import { YieldCurveChart } from "./YieldCurveChart";

// Only states present in the national_processed_v2.csv dataset
const NIGERIA_STATES = ["Kano", "Kebbi", "Niger", "Jigawa", "Ebonyi", "Taraba"];

export default function YieldDashboard() {
  const [selectedState, setSelectedState] = useState("Kano");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="w-full h-full bg-background/95 backdrop-blur-md overflow-hidden pt-6 md:pt-2 pb-2 px-4 md:pl-24 flex items-center justify-center">
      <div className="w-full max-w-[98vw] md:max-w-[98vw] h-[96vh] flex flex-col mx-auto">
        {/* Header */}
        <div className="mb-3 flex flex-col md:flex-row justify-between items-start md:items-end gap-4 shrink-0">
          <div className="space-y-1">
            <h1 className="text-3xl font-extrabold tracking-tight text-foreground flex items-center gap-3">
              <TrendingUp className="w-8 h-8 text-emerald-500" />
              AgroSense Yield Intelligence
            </h1>
            <p className="text-muted-foreground text-sm max-w-xl">
              Deep Learning (LSTM) & XGBoost Ensemble Forecasts. Compare state-level historical yield curves against the regional average benchmark.
            </p>
          </div>

          {/* State Selector Dropdown */}
          <div className="relative">
            <button 
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center justify-between gap-3 bg-card border border-border/50 px-5 py-3 rounded-2xl text-sm font-bold shadow-sm hover:border-emerald-500/50 transition-colors w-[240px]"
            >
              <div className="flex items-center gap-2">
                <MapPin className="w-5 h-5 text-emerald-500" />
                {selectedState} State
              </div>
              <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
            </button>
            
            {dropdownOpen && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute top-full right-0 mt-2 w-[240px] max-h-[350px] overflow-y-auto bg-card border border-border rounded-2xl shadow-2xl z-[500]"
              >
                {NIGERIA_STATES.map(state => (
                  <button
                    key={state}
                    onClick={() => { setSelectedState(state); setDropdownOpen(false); }}
                    className={`w-full text-left px-5 py-3 text-sm font-medium transition-colors hover:bg-muted ${
                      selectedState === state ? "text-emerald-500 bg-emerald-500/10" : "text-foreground"
                    }`}
                  >
                    {state}
                  </button>
                ))}
              </motion.div>
            )}
          </div>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-6 gap-4 grow min-h-0 pt-2">
          
          {/* Main Chart Area */}
          <div className="lg:col-span-5 bg-card border border-border/50 rounded-2xl shadow-xl overflow-hidden flex flex-col min-h-0">
            <div className="p-3 border-b border-border/50 flex items-center gap-2 bg-muted/20 shrink-0">
              <Cpu className="w-5 h-5 text-indigo-500" />
              <h2 className="font-bold text-sm tracking-wide">ENSEMBLE YIELD FORECAST CURVE</h2>
            </div>
            <div className="p-3 grow flex flex-col min-h-0">
               <YieldCurveChart stateName={selectedState} height="100%" />
            </div>
          </div>

          {/* Sidebar Stats Area */}
          <div className="lg:col-span-1 space-y-4 overflow-y-auto pr-1 pb-2">
            <div className="bg-card border border-border/50 rounded-2xl shadow-xl p-4">
              <div className="flex items-center gap-2 mb-4 text-muted-foreground">
                <Leaf className="w-4 h-4" />
                <h3 className="font-bold text-xs uppercase tracking-wider">Model Interpretation</h3>
              </div>
              <div className="space-y-4">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Primary Driver</p>
                  <p className="font-bold text-sm text-foreground">Peak Vegetative NDVI</p>
                  <div className="w-full h-1.5 bg-muted rounded-full mt-2">
                    <div className="h-full bg-emerald-500 rounded-full" style={{ width: '85%' }}></div>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Secondary Driver</p>
                  <p className="font-bold text-sm text-foreground">Cumulative Precipitation</p>
                  <div className="w-full h-1.5 bg-muted rounded-full mt-2">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: '60%' }}></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-emerald-500/20 to-indigo-500/20 border border-emerald-500/30 rounded-2xl shadow-xl p-4 relative overflow-hidden">
              <div className="absolute top-0 right-0 p-4 opacity-10">
                <Cpu className="w-24 h-24" />
              </div>
              <h3 className="font-extrabold text-lg text-foreground mb-1 relative z-10">FastAPI Live Stream</h3>
              <p className="text-xs text-muted-foreground font-medium relative z-10">
                The yield curves are generated securely in real-time via the Python backend combining 10-step temporal LSTM with tabular XGBoost representations.
              </p>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
