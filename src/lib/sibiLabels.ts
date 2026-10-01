// SIBI alphabet labels. Static SIBI covers 24 handshapes: A-I, K-Y.
// J and Z are dynamic (motion) signs, so all static models exclude them.
// (Full A-Z pad remains in the UI for manual simulation / language testing.)

export const SIBI_LABELS_24: string[] = [
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
	'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
	'T', 'U', 'V', 'W', 'X', 'Y'
];

/** Legacy full alphabet (manual sim pad only, since J/Z have no trained model). */
export const SIBI_LABELS: string[] = Array.from({ length: 26 }, (_, i) =>
	String.fromCharCode(65 + i)
);

export const SIBI_INDEX: Record<string, number> = Object.fromEntries(
	SIBI_LABELS_24.map((c, i) => [c, i])
);

export const MODEL_INPUT_SIZE = 128;

export const MODEL_URLS = {
	joint: '/models/sibi-joint/model.json',
	jointBaseline: '/models/sibi-joint-baseline/model.json',
	image: '/models/sibi-image/model.json'
} as const;

/** @deprecated use MODEL_URLS */
export const MODEL_URL = MODEL_URLS.image;
