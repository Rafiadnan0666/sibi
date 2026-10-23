// Check TS normalization math == python v2 canonical normalization.
// Usage: first dump vectors with the sibi-train CSVs present, then run:
//   node scripts/verify-norm.mjs
import { readFileSync } from 'node:fs';

const vecs = JSON.parse(
	readFileSync('C:\\Users\\BRAVO\\AppData\\Local\\Temp\\opencode\\sibi-train\\verify_vectors.json', 'utf8')
);
let maxDiff = 0;
for (let s = 0; s < vecs.baseline.length; s++) {
	const px = vecs.baseline[s].feat; // 63 raw pixel values
	const ref = vecs.joint[s].feat; // python v2 canonical 63
	const P = [];
	for (let i = 0; i < 21; i++) P.push([px[i * 3], px[i * 3 + 1], px[i * 3 + 2]]);
	const w = P[0];
	const mcp = P[9];
	const size = Math.hypot(mcp[0] - w[0], mcp[1] - w[1], mcp[2] - w[2]) + 1e-6;
	const dx = [];
	const dy = [];
	const dz = [];
	for (let i = 0; i < 21; i++) {
		dx.push((P[i][0] - w[0]) / size);
		dy.push((P[i][1] - w[1]) / size);
		dz.push((P[i][2] - w[2]) / size);
	}
	const phi = Math.atan2(dx[9], dy[9]);
	const ca = Math.cos(phi);
	const sa = Math.sin(phi);
	const out = [];
	for (let i = 0; i < 21; i++) {
		out.push(dx[i] * ca - dy[i] * sa, dx[i] * sa + dy[i] * ca, dz[i]);
	}
	if (out[1 * 3] < 0) for (let i = 0; i < 21; i++) out[i * 3] *= -1;
	for (let i = 0; i < 63; i++) {
		maxDiff = Math.max(maxDiff, Math.abs(out[i] - ref[i]));
	}
}
console.log('TS-math vs python-v2 maxAbsDiff =', maxDiff.toExponential(3));
console.log(maxDiff < 1e-6 ? 'NORM-OK' : 'NORM-FAIL');
