// Check TS normalization math == python training normalization.
import { readFileSync } from 'node:fs';
const vecs = JSON.parse(
	readFileSync('C:\\Users\\BRAVO\\AppData\\Local\\Temp\\opencode\\sibi-train\\verify_vectors.json', 'utf8')
);
let maxDiff = 0;
for (let s = 0; s < vecs.baseline.length; s++) {
	const px = vecs.baseline[s].feat; // 63 raw pixel values
	const ref = vecs.joint[s].feat; // python-normalized 63
	const P = [];
	for (let i = 0; i < 21; i++) P.push([px[i * 3], px[i * 3 + 1], px[i * 3 + 2]]);
	const w = P[0];
	const mcp = P[9];
	const size = Math.hypot(mcp[0] - w[0], mcp[1] - w[1], mcp[2] - w[2]) + 1e-6;
	for (let i = 0; i < 21; i++) {
		for (let a = 0; a < 3; a++) {
			const v = (P[i][a] - w[a]) / size;
			maxDiff = Math.max(maxDiff, Math.abs(v - ref[i * 3 + a]));
		}
	}
}
console.log('TS-math vs python maxAbsDiff =', maxDiff.toExponential(3));
console.log(maxDiff < 1e-6 ? 'NORM-OK' : 'NORM-FAIL');
