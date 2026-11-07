/* Memory-access accounting for the wasm linear memory region of rv32emu.
 * Every load/store the RV32 build makes into [LIN, LIN + 32 MB) is recorded at
 * byte granularity: per-phase distinct words written/read (reads split into words
 * never written since boot vs words written after boot), distinct 128-byte lines
 * written, sequential store runs, whole-line overwrite episodes, per-word write
 * counts, and the address extremes the host needs for stack/heap high-water marks.
 */
static uint32_t LIN = 0x02000000u; /* wasm linear memory base (rv_set_lin) */
#define LIN_BYTES 0x02000000u
#define NW (LIN_BYTES / 4)
#define NL (LIN_BYTES / 128)
#define MAXPH 64

enum { F_EVER = 1, F_AFTER_BOOT = 2, F_W_STEP = 4, F_R_STEP = 8, F_BOOT = 16 };
enum { L_OPEN = 1, L_W_STEP = 2, L_EVER = 4, L_FULL_ONCE = 8 };

typedef struct {
  uint64_t icount_start, icount_end;
  uint64_t loads, stores, bytes_loaded, bytes_stored;
  uint64_t words_written, lines_written, words_read_ro, words_read_rw, new_words_written;
} phase_t;

static uint32_t* wcount;
static uint8_t* wflag;
static uint32_t (*lmask)[4];
static uint8_t* lflag;
static phase_t ph[MAXPH];
static int cur_ph;
static int boot_phases = 2; /* phases < boot_phases are "boot" */
static uint64_t run_hist[24];
static uint32_t last_store_end = 0xffffffffu, run_bytes;
static uint64_t episodes, episodes_full, lines_full_once;
static uint32_t min_store = 0xffffffffu, max_store, lo_watch = 0, hi_watch = 0, min_watch_store = 0xffffffffu;
static int mt_on;

static void run_flush(void) {
  if (run_bytes) {
    int b = 0;
    while ((1u << (b + 1)) <= run_bytes && b < 23) b++;
    run_hist[b]++;
  }
  run_bytes = 0;
}

EXPORT int mt_init(void) {
  wcount = calloc(NW, 4);
  wflag = calloc(NW, 1);
  lmask = calloc(NL, 16);
  lflag = calloc(NL, 1);
  mt_on = 1;
  return wcount && wflag && lmask && lflag ? 0 : -1;
}

EXPORT void rv_set_lin(uint32_t base) { LIN = base; }
EXPORT void mt_set_watch(uint32_t lo, uint32_t hi) { lo_watch = lo; hi_watch = hi; }
EXPORT void mt_set_boot_phases(int n) { boot_phases = n; }

static void close_line(uint32_t l) {
  if (lflag[l] & L_OPEN) {
    episodes++;
    if ((lmask[l][0] & lmask[l][1] & lmask[l][2] & lmask[l][3]) == 0xffffffffu) {
      episodes_full++;
      if (!(lflag[l] & L_FULL_ONCE)) { lflag[l] |= L_FULL_ONCE; lines_full_once++; }
    }
    lmask[l][0] = lmask[l][1] = lmask[l][2] = lmask[l][3] = 0;
    lflag[l] &= ~L_OPEN;
  }
}

static inline void mt_load(uint32_t ad, uint32_t n) {
  if (!mt_on || ad < LIN || ad - LIN >= LIN_BYTES) return;
  uint32_t o = ad - LIN;
  phase_t* p = &ph[cur_ph];
  p->loads++;
  p->bytes_loaded += n;
  for (uint32_t w = o >> 2; w <= (o + n - 1) >> 2; w++) {
    if (!(wflag[w] & F_R_STEP)) {
      wflag[w] |= F_R_STEP;
      if (wflag[w] & F_AFTER_BOOT) p->words_read_rw++; else p->words_read_ro++;
    }
  }
  close_line(o >> 7);
  if ((o + n - 1) >> 7 != o >> 7) close_line((o + n - 1) >> 7);
}

static inline void mt_store(uint32_t ad, uint32_t n) {
  if (!mt_on || ad < LIN || ad - LIN >= LIN_BYTES) return;
  uint32_t o = ad - LIN;
  phase_t* p = &ph[cur_ph];
  p->stores++;
  p->bytes_stored += n;
  if (o == last_store_end) run_bytes += n; else { run_flush(); run_bytes = n; }
  last_store_end = o + n;
  if (o < min_store) min_store = o;
  if (o + n > max_store) max_store = o + n;
  if (o >= lo_watch && o < hi_watch && o < min_watch_store) min_watch_store = o;
  int boot = cur_ph < boot_phases;
  for (uint32_t w = o >> 2; w <= (o + n - 1) >> 2; w++) {
    wcount[w]++;
    if (!(wflag[w] & F_EVER)) p->new_words_written++;
    if (!(wflag[w] & F_W_STEP)) { wflag[w] |= F_W_STEP; p->words_written++; }
    wflag[w] |= F_EVER | (boot ? F_BOOT : F_AFTER_BOOT);
  }
  for (uint32_t i = 0; i < n; i++) {
    uint32_t b = o + i, l = b >> 7, k = b & 127;
    lmask[l][k >> 5] |= 1u << (k & 31);
    if (!(lflag[l] & L_W_STEP)) { lflag[l] |= L_W_STEP; p->lines_written++; }
    lflag[l] |= L_OPEN | L_EVER;
  }
}

/* The host's own writes into guest memory (stdin data, tool results, stream
 * chunks): they define words like a store does, but are not CPU stores. */
static uint64_t host_bytes;
EXPORT void mt_host_store(uint32_t ad, uint32_t n) {
  if (nfz) frozen_store_check(ad, n);
  if (!mt_on || ad < LIN || ad - LIN >= LIN_BYTES || n == 0) return;
  uint32_t o = ad - LIN;
  host_bytes += n;
  int boot = cur_ph < boot_phases;
  for (uint32_t w = o >> 2; w <= (o + n - 1) >> 2; w++) {
    wcount[w]++;
    wflag[w] |= F_EVER | (boot ? F_BOOT : F_AFTER_BOOT);
  }
  for (uint32_t b = o; b < o + n; b++) {
    uint32_t l = b >> 7, k = b & 127;
    lmask[l][k >> 5] |= 1u << (k & 31);
    lflag[l] |= L_OPEN | L_EVER;
  }
}

EXPORT void mt_phase(int k) {
  ph[cur_ph].icount_end = mt_icount_now;
  if (!mt_on) { cur_ph = k < MAXPH ? k : MAXPH - 1; ph[cur_ph].icount_start = mt_icount_now; return; }
  for (uint32_t w = 0; w < NW; w++) wflag[w] &= ~(F_W_STEP | F_R_STEP);
  for (uint32_t l = 0; l < NL; l++) lflag[l] &= ~L_W_STEP;
  cur_ph = k < MAXPH ? k : MAXPH - 1;
  ph[cur_ph].icount_start = mt_icount_now;
}

/* Summary: totals over words/lines, histograms. out[] layout documented in rvhost.py. */
EXPORT phase_t* mt_phases(void) { ph[cur_ph].icount_end = mt_icount_now; return ph; }
EXPORT uint64_t* mt_run_hist(void) { run_flush(); return run_hist; }
EXPORT void mt_summary(uint64_t* out) {
  uint64_t ever = 0, boot = 0, after = 0, both = 0, nonzero = 0, lines_ever = 0, open_end = 0;
  uint64_t whist[16] = {0};
  for (uint32_t w = 0; w < NW; w++) {
    uint8_t f = wflag[w];
    if (f & F_EVER) {
      ever++;
      uint32_t v;
      memcpy(&v, M + LIN + 4 * w, 4);
      if (v) nonzero++;
      uint32_t c = wcount[w];
      int b = 0;
      while ((1u << (b + 1)) <= c && b < 15) b++;
      whist[b]++;
    }
    if (f & F_BOOT) boot++;
    if (f & F_AFTER_BOOT) after++;
    if ((f & F_BOOT) && (f & F_AFTER_BOOT)) both++;
  }
  for (uint32_t l = 0; l < NL; l++) {
    if (lflag[l] & L_EVER) lines_ever++;
    if (lflag[l] & L_OPEN) open_end++;
  }
  out[0] = ever; out[1] = boot; out[2] = after; out[3] = both; out[4] = nonzero;
  out[5] = lines_ever; out[6] = episodes; out[7] = episodes_full; out[8] = lines_full_once; out[9] = open_end;
  out[10] = min_store; out[11] = max_store; out[12] = min_watch_store; out[13] = host_bytes;
  for (int i = 0; i < 16; i++) out[16 + i] = whist[i];
}

/* Distinct words ever written, per 64 KB page of linear memory (512 pages). */
EXPORT void mt_page_hist(uint32_t* out) {
  for (uint32_t i = 0; i < 512; i++) out[i] = 0;
  for (uint32_t w = 0; w < NW; w++)
    if (wflag[w] & F_EVER) out[w >> 14]++;
}
