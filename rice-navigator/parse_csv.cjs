const fs = require('fs');
const text = fs.readFileSync('public/nigeria_rice_farms_6_states (1).csv', 'utf8');
const lines = text.trim().split(/\r?\n/);

let output = [];
let idCounter = 1;

lines.slice(1).forEach(line => {
  let result = [];
  let current = '';
  let inQuotes = false;
  for(let i=0; i<line.length; i++) {
    if(line[i] === '"') {
      inQuotes = !inQuotes;
    } else if(line[i] === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += line[i];
    }
  }
  result.push(current);
  
  if (result.length >= 7) {
    const state = result[0];
    const name = result[1];
    const typeStr = result[2];
    const lga = result[4];
    const lat = parseFloat(result[5]);
    const lon = parseFloat(result[6]);
    
    // Assign a random type for the UI
    const types = ["healthy", "stress", "irrigation", "growth"];
    const type = types[Math.floor(Math.random() * types.length)];
    const hectares = Math.floor(Math.random() * 400) + 50;
    
    if (!isNaN(lat) && !isNaN(lon)) {
      output.push(`  { id: "csv_${idCounter++}", name: "${name.replace(/"/g, '\\"')}", state: "${state}", lga: "${lga}", position: [${lat}, ${lon}], type: "${type}", hectares: ${hectares}, details: "${typeStr}" },`);
    }
  }
});

fs.writeFileSync('public/generated_farms.txt', output.join('\n'));
console.log('Done!');
