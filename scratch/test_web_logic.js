const fs = require('fs');

// Read script_1
const content = fs.readFileSync('scratch/check_script_1.js', 'utf8');

// We want to test filename matching
function matchFilename(filename) {
  const CLINICAL_REGISTRY = JSON.parse(content.match(/const CLINICAL_REGISTRY = ({.*?});/)[1]);
  
  let matchedRecord = null;
  const match = filename.match(/^(\d+)\s*[-_]/) || 
                filename.match(/subject[_\s-]*(\d+)/i) || 
                filename.match(/^(\d+)\./) || 
                filename.match(/(\d+)/);

  if (match) {
    const snoCandidate = parseInt(match[1]);
    if (CLINICAL_REGISTRY[snoCandidate]) {
      matchedRecord = CLINICAL_REGISTRY[snoCandidate];
    }
  }
  return matchedRecord;
}

const testFiles = [
  "1 - 0S.jpg",
  "1 - 10S.jpg",
  "2 - 0S.jpg",
  "3 - 0S.jpg",
  "4 - 0S.jpg",
  "10 - 0S.jpg",
  "11 - 0S.jpg",
  "15 - 0S.jpg",
  "45 - 0S.jpg",
  "52 - 0S.jpg",
  "60 - 0S.jpg",
  "68 - 0S.jpg",
  "70 - 0S.jpg",
  "subject_1_0S.jpg",
  "subject_2_0S.jpg"
];

testFiles.forEach(f => {
  const rec = matchFilename(f);
  if (!rec) {
    console.log(`FAIL: Could not match ${f}`);
  } else {
    console.log(`OK: ${f} -> Subject #${rec.sno}: ${rec.overall_category} | RE: ${rec.re.diagnosis}, LE: ${rec.le.diagnosis} (OSDI: ${rec.osdi})`);
  }
});
