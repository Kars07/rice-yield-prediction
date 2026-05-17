import { useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import NigeriaMap from "@/components/NigeriaMap";
import SearchHeader from "@/components/SearchHeader";
import FilterChips from "@/components/FilterChips";
import MapLayersControl, { type MapStyle } from "@/components/MapLayersControl";
import InfoSheet from "@/components/InfoSheet";
import FeatureOverlayCards from "@/components/FeatureOverlayCards";
import BottomNav from "@/components/BottomNav";
import AlertsPage from "@/pages/AlertsPage";
import YieldDashboard from "@/components/YieldDashboard";
import { type RiceField } from "@/data/nigeriaRiceData";
import { type LayerKey } from "@/data/layerSources";
import { type FeatureKey } from "@/data/csvProcessor";
import { useCsvData } from "@/hooks/useCsvData";
import { useRiceFarms } from "@/hooks/useRiceFarms";

const Index = () => {
  const [activeLayer, setActiveLayer] = useState<LayerKey>("healthy");
  const [activeFeature, setActiveFeature] = useState<FeatureKey>("all");
  const [mapStyle, setMapStyle] = useState<"standard" | "satellite" | "terrain">("standard");
  const [activeTab, setActiveTab] = useState("explore");
  const [selectedField, setSelectedField] = useState<RiceField | null>(null);
  const [mapView, setMapView] = useState<"state" | "farm">("farm");
  const [selectedState, setSelectedState] = useState<string | null>(null);
  const [flyTo, setFlyTo] = useState<[number, number] | null>(null);
  const [flyToZoom, setFlyToZoom] = useState<number | undefined>(undefined);

  // Load and process CSV data
  const { metrics, rawRecords } = useCsvData();
  const { farms, loading } = useRiceFarms(metrics);

  const handleFieldClick = useCallback((field: RiceField) => {
    setSelectedField(field);
    setFlyTo(field.position);
    setFlyToZoom(11);
  }, []);

  const handleMapClick = useCallback(() => {
    setSelectedField(null);
    setSelectedState(null);
  }, []);

  const handleStateClick = useCallback((stateName: string) => {
    setSelectedState(stateName);
  }, []);

  const handleSearch = useCallback((_q: string) => {}, []);

  const handleSelectField = useCallback(
    (field: RiceField) => {
      setActiveTab("explore");
      setMapView("farm");
      handleFieldClick(field);
    },
    [handleFieldClick]
  );

  const handleFeatureChange = useCallback((key: FeatureKey) => {
    setActiveFeature(key);
    // Map feature to the closest base layer for the GeoJSON coloring
    const featureToLayer: Partial<Record<FeatureKey, LayerKey>> = {
      healthy_fields:       "healthy",
      ndvi_vigor:           "ndvi",
      growth_trends:        "ndvi",
      temp_stress:          "temperature",
      rainfall_patterns:    "rainfall",
      historical:           "ndvi",
      phenology:            "yield",
      all:                  "healthy",
      irrigation_junctions: "irrigation",
      custom_zones:         "healthy",
      risk_heatmap:         "risk",
      yield_prediction:     "yield",
    };
    setActiveLayer(featureToLayer[key] ?? "healthy");
    setSelectedState(null);
    setSelectedField(null);
  }, []);

  return (
    <div className="h-screen w-screen overflow-hidden relative bg-background">

      {/* MAP — visible if in explore mode or alerts mode (underneath) */}
      <div className={`w-full h-full absolute inset-0 ${activeTab === 'yield' ? 'hidden' : 'block'}`}>
        <NigeriaMap
          mapStyle={mapStyle}
          activeLayer={activeLayer}
          activeFeature={activeFeature}
          csvMetrics={metrics}
          mapView={mapView}
          selectedField={selectedField}
          flyTo={flyTo}
          flyToZoom={flyToZoom}
          riceFields={farms}
          rawRecords={rawRecords}
          onFieldClick={handleFieldClick}
          onMapClick={handleMapClick}
          onStateClick={handleStateClick}
        />
      </div>

      {/* EXPLORE OVERLAY CONTROLS */}
      {activeTab === "explore" && (
        <>
          <SearchHeader 
            riceFields={farms}
            onSearch={handleSearch} 
            onSelectField={handleSelectField} 
          />
          <FilterChips mapView={mapView} onChangeView={setMapView} activeFeature={activeFeature} onChange={handleFeatureChange} />
          <MapLayersControl 
            mapStyle={mapStyle} 
            activeFeature={activeFeature}
            onStyleChange={setMapStyle} 
            onFeatureChange={handleFeatureChange}
          />
          <InfoSheet field={selectedField} csvMetrics={metrics} onClose={() => setSelectedField(null)} />
          {mapView === "state" && (
            <FeatureOverlayCards 
              activeFeature={activeFeature} 
              stateName={selectedState} 
              metrics={selectedState ? metrics?.[selectedState] ?? null : null} 
              onClose={() => setSelectedState(null)} 
            />
          )}
        </>
      )}

      {/* YIELD DASHBOARD */}
      <AnimatePresence>
        {activeTab === "yield" && (
           <motion.div
             initial={{ opacity: 0, scale: 0.98 }}
             animate={{ opacity: 1, scale: 1 }}
             exit={{ opacity: 0, scale: 0.98 }}
             transition={{ duration: 0.3 }}
             className="absolute inset-0 z-[400]"
           >
             <YieldDashboard />
           </motion.div>
        )}
      </AnimatePresence>

      {/* ALERTS OVERLAY (RESPONSIVE) */}
      <AnimatePresence>
        {activeTab === "alerts" && (
          <div className="overflow-y-auto">
            {/* background dim */}
            <motion.div
              className="absolute inset-0 bg-black/40 z-[400]"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setActiveTab("explore")}
            />

            {/* 📱 MOBILE: bottom sheet */}
            <motion.div
              key="alerts-mobile"
              initial={{ y: "100%" }}
              animate={{ y: 0 }}
              exit={{ y: "100%" }}
              transition={{ type: "tween", duration: 0.3 }}
              className="
                absolute bottom-0 left-0 w-full h-[95%]
                bg-background/95 backdrop-blur-md
                z-[500] shadow-2xl
                rounded-t-2xl
                md:hidden
              "
            >
              <AlertsPage 
                metrics={metrics} 
                onViewAlert={(field) => {
                  setActiveTab("explore");
                  handleSelectField(field);
                }} 
                onClose={() => setActiveTab("explore")}
              />
            </motion.div>

            {/* 💻 DESKTOP: left drawer */}
            <motion.div
              key="alerts-desktop"
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "tween", duration: 0.3 }}
              className="
                hidden md:block
                absolute top-0 left-20 h-full
                w-[45%]
                bg-background/90 backdrop-blur-md
                z-[500]
                shadow-2xl
              "
            >
              <AlertsPage 
                metrics={metrics} 
                onViewAlert={(field) => {
                  setActiveTab("explore");
                  handleSelectField(field);
                }} 
                onClose={() => setActiveTab("explore")}
              />
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* BOTTOM NAV */}
      <BottomNav active={activeTab} onChange={setActiveTab} />
    </div>
  );
};

export default Index;