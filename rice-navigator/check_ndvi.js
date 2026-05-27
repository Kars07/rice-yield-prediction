import fs from 'fs';

const text = fs.readFileSync('public/national_processed_v2.csv', 'utf8');
const lines = text.trim().split(/\r?\n/);
const records = [];

for (let i = 1; i < lines.length; i++) {
  const cols = lines[i].split(",");
  if (cols.length < 10) continue;
  const ndvi = parseFloat(cols[1]);
  if (isNaN(ndvi)) continue;
  records.push({
    ndvi,
    state: cols[8].trim(),
    year: parseInt(cols[9])
  });
}

const grouped = {};
for (const r of records) {
  (grouped[r.state] ??= []).push(r);
}

for (const [state, recs] of Object.entries(grouped)) {
  const latestYear = Math.max(...recs.map(r => r.year));
  const latestRecs = recs.filter(r => r.year === latestYear);
  const latest = latestRecs[latestRecs.length - 1];
  console.log(`${state}: latest NDVI is ${latest.ndvi} (Year: ${latestYear})`);
}
