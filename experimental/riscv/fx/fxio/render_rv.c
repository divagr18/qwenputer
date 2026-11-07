/* What an in-weights model server's text work costs on the RV32 core (docs/fx/io.md).
 *
 * Two jobs, on the real bytes of a recorded session, run in rv32emu; render_cost.py checks every
 * output byte against the host's delta.py:
 *
 *   mode 1: an AI Gateway request body -> the template text Qwen does not have yet, for a request
 *           that follows a tool call: <|im_end|>\n, the tool responses, the generation prompt.
 *   mode 2: Qwen's raw completion -> the SSE fx reads (reasoning, text, tool calls, finish).
 *
 * Only the happy path of Qwen's format: no malformed-call recovery, no schema coercion (every
 * parameter of the five tools is a string), no second turn's template. Token ids <-> bytes are
 * not here: decoding is a table lookup, BPE encoding is priced separately in io.md.
 *
 * Mailbox at 0x01F00000: mode, input length, usage prompt tokens, usage completion tokens, call
 * index. Input at 0x02000000, output at 0x04000000; the output length comes back in a0.
 */
typedef unsigned char u8;
typedef unsigned int u32;

#define MB ((volatile u32*)0x01F00000)
static const char* const IN = (const char*)0x02000000;
static char* const OUT = (char*)0x04000000;
static char* o;

__attribute__((naked, section(".text.start"))) void _start(void) {
  __asm__ volatile("li sp, 0x07F00000\n call rv_main\n li a7, 93\n ecall\n1: j 1b\n");
}

static void puts_(const char* s) { while (*s) *o++ = *s++; }
static void mem(const char* s, u32 n) { for (u32 i = 0; i < n; i++) o[i] = s[i]; o += n; }
static void num(u32 v) {
  char b[12]; int n = 0;
  do { b[n++] = '0' + v % 10; v /= 10; } while (v);
  while (n) *o++ = b[--n];
}
static int starts(const char* p, const char* e, const char* s) {
  while (*s) { if (p >= e || *p++ != *s++) return 0; }
  return 1;
}
static const char* find(const char* p, const char* e, const char* s) {
  for (; p < e; p++) if (*p == *s && starts(p, e, s)) return p;
  return 0;
}

/* ---- JSON, as much as walking fx's request takes ---- */
static const char* ws(const char* p) { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++; return p; }
static const char* skip_str(const char* p) { p++; while (*p != '"') p += *p == '\\' ? 2 : 1; return p + 1; }
static const char* skip(const char* p) {
  p = ws(p);
  if (*p == '"') return skip_str(p);
  if (*p == '{' || *p == '[') {
    int d = 0;
    do {
      if (*p == '"') { p = skip_str(p); continue; }
      if (*p == '{' || *p == '[') d++;
      if (*p == '}' || *p == ']') d--;
      p++;
    } while (d);
    return p;
  }
  while (*p != ',' && *p != '}' && *p != ']') p++;
  return p;
}
/* The value of a key of the object at p, or 0. */
static const char* field(const char* p, const char* key) {
  p = ws(p) + 1;
  for (;;) {
    p = ws(p);
    if (*p == '}') return 0;
    const char* k = p + 1;
    p = skip_str(p);
    int hit = 1;
    const char* s = key, *q = k;
    while (*s) if (*q++ != *s++) { hit = 0; break; }
    hit = hit && *q == '"';
    p = ws(p) + 1;
    if (hit) return ws(p);
    p = ws(skip(p));
    if (*p == ',') p++;
  }
}
static u32 hex4(const char* p) {
  u32 v = 0;
  for (int i = 0; i < 4; i++) { char c = p[i]; v = v * 16 + (c <= '9' ? c - '0' : (c | 32) - 'a' + 10); }
  return v;
}
/* Decodes the JSON string at p into o; returns the end of the decoded text. */
static char* unescape(const char* p) {
  p++;
  while (*p != '"') {
    if (*p != '\\') { *o++ = *p++; continue; }
    char c = p[1];
    p += 2;
    if (c == 'n') *o++ = '\n';
    else if (c == 't') *o++ = '\t';
    else if (c == 'r') *o++ = '\r';
    else if (c == 'b') *o++ = 8;
    else if (c == 'f') *o++ = 12;
    else if (c == 'u') {
      u32 v = hex4(p);
      p += 4;
      if (v < 0x80) *o++ = v;
      else if (v < 0x800) { *o++ = 0xC0 | v >> 6; *o++ = 0x80 | (v & 63); }
      else { *o++ = 0xE0 | v >> 12; *o++ = 0x80 | ((v >> 6) & 63); *o++ = 0x80 | (v & 63); }
    } else *o++ = c;
  }
  return o;
}
static int is_ws(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

static void tool_message(const char* m) {
  const char* c = ws(field(m, "content")) + 1;
  for (;;) {
    c = ws(c);
    if (*c == ']') break;
    const char* out = field(field(c, "output"), "value");
    puts_("\n<tool_response>\n");
    char* a = o;
    unescape(out);
    char* e = o;  /* the template trims the content */
    while (e > a && is_ws(e[-1])) e--;
    char* s = a;
    while (s < e && is_ws(*s)) s++;
    if (s != a) { u32 n = e - s; for (u32 i = 0; i < n; i++) a[i] = s[i]; e = a + n; }
    o = e;
    puts_("\n</tool_response>");
    c = ws(skip(c));
    if (*c == ',') c++;
  }
}

static u32 render(void) {
  const char* m = field(IN, "prompt");
  const char* tail = 0;  /* the first tool message after the last assistant message */
  const char* p = ws(m + 1);
  while (*p != ']') {
    const char* r = field(p, "role");
    if (starts(r, r + 12, "\"assistant\"")) tail = 0;
    else if (starts(r, r + 7, "\"tool\"") && !tail) tail = p;
    p = ws(skip(p));
    if (*p == ',') p = ws(p + 1);
  }
  puts_("<|im_end|>\n<|im_start|>user");
  while (*tail != ']') {
    tool_message(tail);
    tail = ws(skip(tail));
    if (*tail == ',') tail = ws(tail + 1);
  }
  puts_("<|im_end|>\n<|im_start|>assistant\n<think>\n");
  return o - OUT;
}

/* ---- Qwen's text -> SSE ---- */
static void esc(const char* s, u32 n) {
  for (u32 i = 0; i < n; i++) {
    u8 c = s[i];
    if (c == '"' || c == '\\') { *o++ = '\\'; *o++ = c; }
    else if (c == '\n') { *o++ = '\\'; *o++ = 'n'; }
    else if (c == '\t') { *o++ = '\\'; *o++ = 't'; }
    else if (c == '\r') { *o++ = '\\'; *o++ = 'r'; }
    else if (c < 32) { puts_("\\u00"); *o++ = '0' + (c >> 4); *o++ = "0123456789abcdef"[c & 15]; }
    else *o++ = c;
  }
}
/* json.dumps with its default separators inside a JSON string: quotes and backslashes doubled */
static void esc_twice(const char* s, u32 n) {
  char* a = o;
  esc(s, n);
  u32 len = o - a;
  static char tmp[1 << 20];
  for (u32 i = 0; i < len; i++) tmp[i] = a[i];
  o = a;
  esc(tmp, len);
}
static void tool_event(const char* s, const char* e, u32 idx, u32 k) {
  const char* f = find(s, e, "<function=") + 10;
  const char* g = find(f, e, ">");
  puts_("data: {\"type\":\"tool-call\",\"toolCallId\":\"call_"); num(idx); *o++ = '_'; num(k);
  puts_("\",\"toolName\":\""); mem(f, g - f); puts_("\",\"input\":\"{");
  const char* p = g;
  int n = 0;
  for (;;) {
    const char* a = find(p, e, "<parameter=");
    if (!a) break;
    a += 11;
    const char* b = find(a, e, ">");
    const char* v = b + 1;
    const char* w = find(v, e, "</parameter>");
    p = w + 12;
    if (n++) puts_(", ");
    puts_("\\\""); mem(a, b - a); puts_("\\\": \\\"");
    if (v < w && *v == '\n') v++;
    if (w > v && w[-1] == '\n') w--;
    esc_twice(v, w - v);
    puts_("\\\"");
  }
  puts_("}\"}\n\n");
}
static void delta_event(const char* kind, const char* id, const char* s, u32 n) {
  puts_("data: {\"type\":\""); puts_(kind); puts_("-start\",\"id\":\""); puts_(id); puts_("\"}\n\n");
  puts_("data: {\"type\":\""); puts_(kind); puts_("-delta\",\"id\":\""); puts_(id); puts_("\",\"delta\":\"");
  esc(s, n);
  puts_("\"}\n\n");
  puts_("data: {\"type\":\""); puts_(kind); puts_("-end\",\"id\":\""); puts_(id); puts_("\"}\n\n");
}

static u32 reply(u32 len, u32 ptok, u32 ctok, u32 idx) {
  const char* e = IN + len;
  const char* t = find(IN, e, "</think>");
  delta_event("reasoning", "r0", IN, t - IN);
  const char* c = t + 8;
  const char* call = find(c, e, "<tool_call>");
  const char* ce = call ? call : e;
  const char* a = c;
  while (a < ce && is_ws(*a)) a++;
  if (a < ce) delta_event("text", "t0", c, ce - c);  /* as vLLM: the framing newlines stay */
  u32 k = 0;
  while (call) {
    const char* end = find(call, e, "</tool_call>");
    tool_event(call, end, idx, k++);
    call = find(end, e, "<tool_call>");
  }
  const char* why = k ? "tool-calls" : "stop";
  puts_("data: {\"type\":\"finish\",\"finishReason\":{\"unified\":\""); puts_(why); puts_("\",\"raw\":\""); puts_(why);
  puts_("\"},\"usage\":{\"inputTokens\":{\"total\":"); num(ptok); puts_("},\"outputTokens\":{\"total\":"); num(ctok);
  puts_("}}}\n\n");
  return o - OUT;
}

u32 rv_main(void) {
  o = OUT;
  return MB[0] == 1 ? render() : reply(MB[1], MB[2], MB[3], MB[4]);
}
