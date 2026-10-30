// Dynamic SIBI letters J and Z.
//
// The static classifier only knows 24 handshapes (A-I, K-Y) because J and Z
// are motion signs: J traces a hook with the pinky (I handshape), Z traces a
// zig-zag with the index finger (D handshape). This module recognises those
// traces from the fingertip path. It is intentionally conservative: it only
// runs while the matching handshape is held, and needs a large, complete
// stroke before it fires, so normal static letters never become J/Z.

export interface TrailPoint {
	x: number;
	y: number;
	t: number;
}

export interface TrailSegment {
	dx: number;
	dy: number;
}

/** Push a fingertip sample into a rolling trail (keeps the last `maxMs`). */
export function pushTrail(
	trail: TrailPoint[],
	p: TrailPoint,
	maxMs = 1800,
	maxPoints = 140
): void {
	trail.push(p);
	while (
		trail.length > 2 &&
		(p.t - trail[0].t > maxMs || trail.length > maxPoints)
	) {
		trail.shift();
	}
}

/**
 * Split the trail into straight strokes. Consecutive samples whose direction
 * stays within one 45-degree bucket are merged; a clear turn starts a new
 * stroke. Tiny jitter strokes are discarded.
 */
export function trailSegments(trail: TrailPoint[]): TrailSegment[] {
	if (trail.length < 3) return [];
	const segs: TrailSegment[] = [];
	let sx = trail[0].x;
	let sy = trail[0].y;
	let px = sx;
	let py = sy;
	let dir = -1;
	for (let i = 1; i < trail.length; i++) {
		const p = trail[i];
		const dx = p.x - px;
		const dy = p.y - py;
		if (Math.hypot(dx, dy) < 0.0025) continue; // ignore sensor jitter
		const d = ((Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8;
		if (dir === -1) {
			dir = d;
		} else {
			const diff = Math.min((d - dir + 8) % 8, (dir - d + 8) % 8);
			if (diff > 1) {
				segs.push({ dx: px - sx, dy: py - sy });
				sx = px;
				sy = py;
				dir = d;
			}
		}
		px = p.x;
		py = p.y;
	}
	segs.push({ dx: px - sx, dy: py - sy });
	return segs.filter((s) => Math.hypot(s.dx, s.dy) > 0.02);
}

/** Distance travelled within the last `ms` milliseconds (jitter included). */
export function trailTravel(trail: TrailPoint[], now: number, ms = 350): number {
	let total = 0;
	for (let i = 1; i < trail.length; i++) {
		if (now - trail[i].t > ms) continue;
		total += Math.hypot(trail[i].x - trail[i - 1].x, trail[i].y - trail[i - 1].y);
	}
	return total;
}

/** Net displacement between the oldest and newest recent trail point. */
export function trailNet(trail: TrailPoint[], now: number, ms = 350): number {
	let first: TrailPoint | null = null;
	let last: TrailPoint | null = null;
	for (const p of trail) {
		if (now - p.t > ms) continue;
		if (!first) first = p;
		last = p;
	}
	return first && last ? Math.hypot(last.x - first.x, last.y - first.y) : 0;
}

/** Z: horizontal stroke, diagonal back the other way, horizontal again. */
export function detectZ(trail: TrailPoint[]): boolean {
	if (trail.length < 8) return false;
	const span = trail[trail.length - 1].t - trail[0].t;
	if (span < 280 || span > 1800) return false;
	const segs = trailSegments(trail);
	for (let i = 0; i + 2 < segs.length; i++) {
		const a = segs[i];
		const b = segs[i + 1];
		const c = segs[i + 2];
		const aHorizontal = Math.abs(a.dx) > 0.05 && Math.abs(a.dx) > Math.abs(a.dy) * 1.2;
		if (!aHorizontal) continue;
		const goRight = a.dx > 0;
		// Middle stroke: downwards and back toward the starting side.
		const bDiagonal =
			b.dy > 0.035 &&
			Math.abs(b.dx) > 0.02 &&
			(b.dx > 0 ? !goRight : goRight);
		if (!bDiagonal) continue;
		// Last stroke: horizontal again in the original direction.
		const cHorizontal =
			Math.abs(c.dx) > 0.03 &&
			(c.dx > 0) === goRight &&
			Math.abs(c.dx) > Math.abs(c.dy) * 0.8;
		if (cHorizontal) return true;
	}
	return false;
}

/** J: long downward stroke followed by a horizontal hook at the bottom. */
export function detectJ(trail: TrailPoint[]): boolean {
	if (trail.length < 8) return false;
	const span = trail[trail.length - 1].t - trail[0].t;
	if (span < 280 || span > 1800) return false;
	const segs = trailSegments(trail);
	for (let i = 0; i + 1 < segs.length; i++) {
		const down = segs[i];
		const hook = segs[i + 1];
		const isDown = down.dy > 0.09 && down.dy > Math.abs(down.dx) * 1.2;
		if (!isDown) continue;
		const isHook =
			Math.abs(hook.dx) > 0.05 && Math.abs(hook.dx) > Math.abs(hook.dy) * 1.4;
		if (isHook) return true;
	}
	return false;
}
