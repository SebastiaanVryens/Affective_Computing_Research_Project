/**
 * IndexedDB persistence for the diary.
 *
 * Everything personal lives here, in the browser, and never goes anywhere else.
 * The backend is stateless by design — it receives audio, returns numbers, and
 * forgets. That's the whole privacy story, and it's worth stating plainly in
 * the ethics section of the report.
 *
 * Two stores:
 *   entries  — one per recorded session, keyed by id, indexed by day
 *   world    — the accumulated 3D world state, a single row
 */

import { type Emotion, type EmotionVector, fromArray, zeroVector } from '../emotions';
import type { Keyword, ModalityReading } from '../api';

const DB_NAME = 'mindscape';
const DB_VERSION = 1;
const ENTRY_STORE = 'entries';
const WORLD_STORE = 'world';

export interface DiaryEntry {
  id: string;
  /** ISO timestamp of when the session started. */
  createdAt: string;
  /** Local YYYY-MM-DD. Indexed, and the unit the history view groups by. */
  day: string;
  durationSeconds: number;
  transcript: string;
  /** The fused reading for the whole entry. */
  vector: EmotionVector;
  dominant: Emotion;
  sentiment: string;
  certainty: number;
  keywords: Keyword[];
  /** Per-modality breakdown, kept for the diagnostics view and for analysis. */
  modalities: ModalityReading[];
  /** The entry's emotional peak sentence, the core-memory candidate. */
  peak: { text: string; emotion: Emotion; score: number } | null;
  /**
   * Which emotions were felt together, and whether the channels corroborated it.
   * Absent on entries saved before blend detection existed, so always optional.
   */
  blend?: {
    /** Non-neutral mass — how much was happening, regardless of how many. */
    charge: number;
    /** Effective number of emotions, 1 = single, 2 = two held at once. */
    complexity: number;
    components: Array<{ emotion: Emotion; share: number }>;
    /** True only when 2+ emotions AND the channels agreed it was real. */
    isBlend: boolean;
    label: string;
  };
  /**
   * How loudly this entry was actually spoken. Absent on entries saved before
   * this existed, and on entries with no speech at all.
   *
   * Stored raw rather than normalised, deliberately: absolute microphone levels
   * are incomparable across machines, so "loud" is only meaningful relative to
   * the rest of *this* diary — a relationship that changes as the diary grows.
   * Normalising at write time would freeze each entry against whatever history
   * happened to exist that day. `vocalRanks()` does it at read time instead.
   */
  vocals?: {
    /** Mean level while speaking, 0-1 after the root-curve compression. */
    mean: number;
    /** Loudest moment in the entry. */
    peak: number;
    /** Spectral centroid — a raised, tense voice sits higher than a calm one. */
    brightness: number;
    /** Roughly how long there was actual speech, in seconds. */
    speakingSeconds: number;
  };
  isCoreMemory: boolean;
  /** User's own words about why this mattered, if they promoted it. */
  note?: string;
  /** Stable position in the 3D world, assigned once and never recomputed. */
  worldPosition: { x: number; y: number; z: number };
}

export interface WorldState {
  id: 'singleton';
  /** Total accumulated emotion mass across all entries — tints the world. */
  lifetimeTotals: EmotionVector;
  entryCount: number;
  /** Monotonic counter used to place each new orb deterministically. */
  placementSeed: number;
  firstEntryAt: string | null;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(ENTRY_STORE)) {
        const store = db.createObjectStore(ENTRY_STORE, { keyPath: 'id' });
        store.createIndex('day', 'day', { unique: false });
        store.createIndex('createdAt', 'createdAt', { unique: false });
        store.createIndex('isCoreMemory', 'isCoreMemory', { unique: false });
      }
      if (!db.objectStoreNames.contains(WORLD_STORE)) {
        db.createObjectStore(WORLD_STORE, { keyPath: 'id' });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  return dbPromise;
}

function transact<T>(
  store: string,
  mode: IDBTransactionMode,
  work: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(store, mode);
        const request = work(tx.objectStore(store));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      })
  );
}

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------

export async function saveEntry(entry: DiaryEntry): Promise<void> {
  await transact(ENTRY_STORE, 'readwrite', (store) => store.put(entry));
}

export async function getEntry(id: string): Promise<DiaryEntry | undefined> {
  return transact<DiaryEntry | undefined>(ENTRY_STORE, 'readonly', (store) =>
    store.get(id)
  );
}

export async function allEntries(): Promise<DiaryEntry[]> {
  const entries = await transact<DiaryEntry[]>(ENTRY_STORE, 'readonly', (store) =>
    store.getAll()
  );
  return entries.sort((a, b) => a.createdAt.localeCompare(b.createdAt));
}

export async function coreMemories(): Promise<DiaryEntry[]> {
  const entries = await allEntries();
  return entries.filter((e) => e.isCoreMemory);
}

export async function deleteEntry(id: string): Promise<void> {
  await transact(ENTRY_STORE, 'readwrite', (store) => store.delete(id));
}

/**
 * Promote or demote an entry as a core memory.
 *
 * Deliberately a user action rather than an automatic threshold. The app can
 * guess which moment was most intense, but only the person living it knows
 * which one was *formative* — and quietly deciding that on their behalf would
 * be the wrong kind of paternalism for a diary.
 */
export async function setCoreMemory(
  id: string,
  isCore: boolean,
  note?: string
): Promise<DiaryEntry | undefined> {
  const entry = await getEntry(id);
  if (!entry) return undefined;
  entry.isCoreMemory = isCore;
  if (note !== undefined) entry.note = note;
  await saveEntry(entry);
  return entry;
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

const EMPTY_WORLD: WorldState = {
  id: 'singleton',
  lifetimeTotals: zeroVector(),
  entryCount: 0,
  placementSeed: 0,
  firstEntryAt: null,
};

export async function loadWorld(): Promise<WorldState> {
  const stored = await transact<WorldState | undefined>(
    WORLD_STORE,
    'readonly',
    (store) => store.get('singleton')
  );
  return stored ?? { ...EMPTY_WORLD, lifetimeTotals: zeroVector() };
}

export async function saveWorld(state: WorldState): Promise<void> {
  await transact(WORLD_STORE, 'readwrite', (store) => store.put(state));
}

/**
 * Fold a new entry into the world's lifetime totals.
 *
 * Contributions are weighted by certainty, so a confidently-sad entry moves the
 * world further than an ambiguous one. This is what makes the world accumulate
 * a personality instead of drifting toward grey as entries pile up.
 */
export function accumulate(world: WorldState, entry: DiaryEntry): WorldState {
  const totals = { ...world.lifetimeTotals };
  const weight = 0.4 + 0.6 * entry.certainty;
  for (const key of Object.keys(totals) as Emotion[]) {
    totals[key] += entry.vector[key] * weight;
  }
  return {
    ...world,
    lifetimeTotals: totals,
    entryCount: world.entryCount + 1,
    placementSeed: world.placementSeed + 1,
    firstEntryAt: world.firstEntryAt ?? entry.createdAt,
  };
}

// ---------------------------------------------------------------------------
// Export / reset
// ---------------------------------------------------------------------------

/** Full JSON dump — the user's data is theirs to take. */
export async function exportAll(): Promise<string> {
  const [entries, world] = await Promise.all([allEntries(), loadWorld()]);
  return JSON.stringify(
    { exportedAt: new Date().toISOString(), version: DB_VERSION, world, entries },
    null,
    2
  );
}

export async function importAll(json: string): Promise<number> {
  const parsed = JSON.parse(json) as { entries?: DiaryEntry[]; world?: WorldState };
  if (!Array.isArray(parsed.entries)) throw new Error('No entries array in import');

  for (const entry of parsed.entries) {
    // Vectors may arrive as arrays from an older export; normalise the shape.
    if (Array.isArray(entry.vector)) {
      entry.vector = fromArray(entry.vector as unknown as number[]);
    }
    await saveEntry(entry);
  }
  if (parsed.world) await saveWorld({ ...parsed.world, id: 'singleton' });
  return parsed.entries.length;
}

export async function wipe(): Promise<void> {
  await transact(ENTRY_STORE, 'readwrite', (store) => store.clear());
  await transact(WORLD_STORE, 'readwrite', (store) => store.clear());
}

/** Local calendar day, not UTC — a 1am entry belongs to the day you felt it. */
export function dayKey(date: Date = new Date()): string {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, '0');
  const day = `${date.getDate()}`.padStart(2, '0');
  return `${year}-${month}-${day}`;
}
