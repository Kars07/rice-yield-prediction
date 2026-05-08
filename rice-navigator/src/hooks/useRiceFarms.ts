import { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { type RiceField } from '@/data/nigeriaRiceData';
import { type StateMetrics } from '@/data/csvProcessor';

// Generate a deterministic pseudo-random number based on string
const hashString = (str: string) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
};

export function useRiceFarms(metrics: Record<string, StateMetrics> | null) {
  const [farms, setFarms] = useState<RiceField[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Papa.parse('/nigeria_rice_farms_6_states.csv', {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const parsedFarms: RiceField[] = results.data.map((row: any, index: number) => {
          const lat = parseFloat(row.Latitude);
          const lng = parseFloat(row.Longitude);
          
          // Realistic indicator logic:
          // Based on state's overall health and some local variance
          let fieldType: RiceField["type"] = "healthy";
          
          if (metrics && metrics[row.State]) {
            const stateMetric = metrics[row.State];
            const variance = hashString(row.Farm_Name || "") % 100;
            
            // If state has temperature stress and variance < 40%, field is stressed
            if (stateMetric.hasTemperatureStress && variance < 40) {
              fieldType = "stress";
            } 
            // If state is growing (phenology) and variance < 50%, field is growing
            else if (stateMetric.phenologyStage === "Tillering" || stateMetric.phenologyStage === "Vegetative") {
              fieldType = variance < 60 ? "growth" : "healthy";
            } 
            // If state is stressed, higher chance of stress
            else if (stateMetric.healthStatus === "Stressed") {
              fieldType = variance < 70 ? "stress" : "irrigation";
            } 
            // Default random healthy/irrigation mixture
            else {
              if (variance < 15) fieldType = "stress";
              else if (variance < 40) fieldType = "irrigation";
              else if (variance < 70) fieldType = "growth";
              else fieldType = "healthy";
            }
          } else {
            // Fallback if metrics not loaded yet
            const v = hashString(row.Farm_Name || "") % 4;
            fieldType = ["healthy", "growth", "irrigation", "stress"][v] as RiceField["type"];
          }

          // Generate realistic hectares based on farm type string
          let hectares = 150 + (hashString(row.Farm_Name || "") % 300);
          if (row.Type && row.Type.includes("Mill")) hectares = 50 + (hashString(row.Farm_Name || "") % 100);
          if (row.Type && row.Type.includes("Cooperative")) hectares = 300 + (hashString(row.Farm_Name || "") % 500);

          return {
            id: `csv_dyn_${index}`,
            name: row.Farm_Name || "Unknown Farm",
            state: row.State || "Unknown State",
            lga: row.LGA || row.Location || "Unknown LGA",
            position: [lat, lng],
            type: fieldType,
            hectares: hectares,
            details: row.Type || "Rice Farm"
          };
        });

        // Filter out invalid coordinates
        const validFarms = parsedFarms.filter(f => !isNaN(f.position[0]) && !isNaN(f.position[1]));
        setFarms(validFarms);
        setLoading(false);
      },
      error: (error) => {
        console.error("Error parsing farms CSV:", error);
        setLoading(false);
      }
    });
  }, [metrics]);

  return { farms, loading };
}
