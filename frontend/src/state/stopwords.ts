/**
 * Display-time filter for keywords that are words rather than themes.
 *
 * Two reasons this exists on the front end, rather than only in the extractor
 * that produced the keywords.
 *
 * The first is that the backend has two extractors and they are not equally
 * good. When YAKE is installed it does its own statistical filtering and
 * "something", "everything" and "expected" never survive it. When it isn't, the
 * built-in RAKE-style fallback in backend/app/keywords.py ranks candidates by
 * phrase length and repetition alone — it has no notion of how *ordinary* a word
 * is, so the words a person says in every single entry rise straight to the top
 * of a list labelled "recurring themes". They are recurring. They are not
 * themes. That extractor's stoplist has been widened to match this file, but
 * widening it only helps entries recorded from now on.
 *
 * Which is the second reason: keywords are extracted once and stored in the
 * entry (see db.ts), so a diary already recorded carries whatever its extractor
 * thought at the time, and no backend change reaches it. Filtering at display
 * time is what fixes the report someone already has.
 *
 * The rule is the same one the backend uses: a phrase survives if *any* of its
 * words carries content. "supervisor meeting" survives, "say anything" does not.
 * That keeps genuine two-word themes intact while dropping the scaffolding that
 * spoken diary entries are mostly made of.
 *
 * The list is a complete mirror rather than a supplement — it repeats the plain
 * function words the backend has always filtered, instead of only the ones it
 * recently gained. That is deliberate: "because" has been stoplisted on the
 * backend since the first commit and still reached a stored entry, by a route
 * that no longer exists in the code. Whatever it was, a filter that assumes the
 * backend caught the basics would have let it through again.
 *
 * Erring: this list is deliberately conservative about anything that could be
 * someone's actual subject. "alone", "tired", "sleep", "money", "night" are all
 * things a diary is about, so none of them are here — a little noise in the
 * table is a far cheaper mistake than silently deleting the one word that
 * mattered.
 */

/**
 * Words that carry no subject on their own.
 *
 * Grouped by why they leak, because the groups are what make it reviewable —
 * a flat alphabetical list of 200 words is impossible to audit and impossible
 * to extend without duplicating half of it.
 */
const LOW_CONTENT = new Set<string>(
  `
  a about above after again against all am an and any are as at be because been
  before being below between both but by can cannot could did do does doing down
  during each few for from further had has have having he her here hers herself
  him himself his how if in into is it its itself let me my myself nor not of off
  on once only or other ought our ours ourselves out over own she so some such
  than that the their theirs them themselves then there these they this those
  through to too under up us we were what when where which while who whom why
  with you your yours yourself yourselves

  like okay ok yeah yep nope uh um erm hmm mm ah eh
  today tomorrow yesterday day days week month year time

  something anything nothing everything someone somebody anyone anybody everyone
  everybody nobody none everywhere anywhere somewhere nowhere

  think thinks thinking thought thoughts feel feels feeling feelings felt know
  knows knowing knew want wants wanted need needs needed seem seems seemed look
  looks looked looking make makes made making say says said saying tell tells
  told telling going went gone come comes coming came take takes taking took
  taken give gives giving gave gets getting put puts putting keep keeps keeping
  kept try tries trying tried use uses using used happen happens happening
  happened start starts starting started stop stops stopping stopped spend
  spends spending spent expect expects expecting expected guess guessed suppose
  supposed wonder wondered mean means meant

  way ways thing things stuff lot lots bit part parts point points moment
  moments morning afternoon evening tonight hour hours minute minutes end kind
  sort type

  really actually basically literally probably definitely maybe perhaps quite
  rather pretty very much many more most less least enough almost nearly just
  properly exactly completely absolutely entirely somehow ordinary person people
  even still ever never always sometimes often usually again already anyway
  though although since while until without around back away whatever honestly
  obviously apparently

  good bad better worse best worst fine okay nice great big small long longer
  short hard easy new old same different whole real sure right wrong able weird
  strange

  will would can could shall should may might must let lets gonna wanna gotta

  because well yes whether else instead afterwards anymore
  `
    .trim()
    .split(/\s+/)
);

/**
 * Whether a keyword is worth showing as a theme.
 *
 * Mirrors backend `_acceptable`: any word with content keeps the phrase. Short
 * tokens are treated as contentless for the same reason the backend does —
 * "an", "ok", "so" surviving inside a bigram tells a reader nothing.
 */
export function isThemeworthy(text: string): boolean {
  const words = text.trim().toLowerCase().split(/\s+/);
  return words.some((word) => word.length >= 3 && !LOW_CONTENT.has(word));
}
