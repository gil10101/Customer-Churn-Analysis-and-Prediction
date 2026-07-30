#!/usr/bin/env python3
"""Builds five static HTML dashboards from results.json into dashboards/.
Self-contained inline SVG, shared warm-neutral style, hover tooltips,
table view under every chart."""
import html as html_mod
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "dashboards")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))

# palette (validated: adjacent-forms order blue,salmon,violet,gold)
PAGE = "#f4f2ee"
SURFACE = "#fafaf9"
BLUE, SALMON, VIOLET, GOLD = "#3a6fa5", "#dd7f68", "#7a68b8", "#d99a3d"
GRAY = "#b5b1a9"
NAVY = "#2e5a80"
INK, INK2, MUTED = "#2b2a28", "#55534e", "#8a877f"
GRID, BASE = "#e7e5df", "#c9c6bd"

SEG_COLORS = {"New · High spend": SALMON, "New · Basic service": GOLD,
              "Established · Premium": BLUE, "Loyal · Value plans": VIOLET}
SEG_ORDER = ["New · High spend", "New · Basic service",
             "Established · Premium", "Loyal · Value plans"]

esc = html_mod.escape


def fmt_n(x):
    return f"{x:,.0f}"


def pct(x, d=1):
    return f"{x*100:.{d}f}%"


def money_k(x):
    sign = "−" if x < 0 else ""
    v = abs(x) / 1000
    return f"{sign}${v:.1f}K" if 0 < v < 10 else f"{sign}${v:.0f}K"


def nice_axis(vmax):
    """Round axis top + step so every tick label is clean."""
    raw = vmax / 4
    mag = 10 ** np.floor(np.log10(raw))
    step = mag
    for m in [1, 2, 2.5, 5, 10]:
        step = m * mag
        if step >= raw:
            break
    top = float(np.ceil(vmax / step) * step)
    return top, float(step), int(round(top / step))


# ----------------------------------------------------------------- svg helpers
def svg_open(w, h):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'style="display:block;font-family:inherit">')


def txt(x, y, s, size=12, color=INK2, anchor="start", weight="400", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{esc(str(s))}</text>')


def rrect_right(x, y, w, h, r, fill, tip=""):
    """Horizontal bar: rounded data-end (right), square baseline (left)."""
    r = min(r, w / 2, h / 2)
    d = (f"M{x:.1f},{y:.1f} h{w - r:.1f} q{r},0 {r},{r} v{h - 2*r:.1f} "
         f"q0,{r} -{r},{r} h-{w - r:.1f} z")
    t = f' data-tip="{esc(tip)}"' if tip else ""
    return f'<path d="{d}" fill="{fill}"{t}/>'


def rrect_top(x, y, w, h, r, fill, tip=""):
    """Column: rounded data-end (top), square baseline (bottom)."""
    r = min(r, w / 2, h / 2)
    d = (f"M{x:.1f},{y + h:.1f} v-{h - r:.1f} q0,-{r} {r},-{r} h{w - 2*r:.1f} "
         f"q{r},0 {r},{r} v{h - r:.1f} z")
    t = f' data-tip="{esc(tip)}"' if tip else ""
    return f'<path d="{d}" fill="{fill}"{t}/>'


def barh_chart(items, w=620, xmax=None, ref=None, ref_label="", val_fmt=None,
               bar_h=20, gap=14, left=170, suffix_fmt=None):
    """items: (label, value, color, tip)."""
    val_fmt = val_fmt or (lambda v: pct(v))
    n = len(items)
    top, bottom = 20 if ref is not None else 8, 26
    h = top + n * (bar_h + gap) - gap + bottom
    plot_w = w - left - 96
    xmax, xstep, nt = nice_axis(xmax or max(v for _, v, _, _ in items) * 1.12)
    parts = [svg_open(w, h)]
    for i in range(nt + 1):
        gx = left + plot_w * i * xstep / xmax
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" '
                     f'y2="{h - bottom + 4}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(txt(gx, h - 8, val_fmt(i * xstep), 11, MUTED, "middle",
                         extra='style="font-variant-numeric:tabular-nums"'))
    for i, (lab, v, color, tip) in enumerate(items):
        y = top + i * (bar_h + gap)
        bw = max(2.5, plot_w * v / xmax)
        parts.append(txt(left - 10, y + bar_h / 2 + 4, lab, 12, INK2, "end"))
        parts.append(rrect_right(left, y, bw, bar_h, 4, color, tip))
        sfx = suffix_fmt(v, i) if suffix_fmt else val_fmt(v)
        parts.append(txt(left + bw + 8, y + bar_h / 2 + 4, sfx, 12, INK,
                         weight="600"))
    if ref is not None:
        rx = left + plot_w * ref / xmax
        parts.append(f'<line x1="{rx:.1f}" y1="{top - 4}" x2="{rx:.1f}" '
                     f'y2="{h - bottom + 4}" stroke="{NAVY}" stroke-width="1.4"/>')
        parts.append(txt(rx + 6, top - 8, ref_label, 11, NAVY))
    parts.append("</svg>")
    return "".join(parts)


def column_chart(items, w=620, h=300, ymax=None, val_fmt=None, unit=""):
    """items: (label, value, color, tip)."""
    val_fmt = val_fmt or (lambda v: pct(v))
    left, right, top, bottom = 64, 16, 14, 46
    plot_w, plot_h = w - left - right, h - top - bottom
    ymax, ystep, nt = nice_axis(ymax or max(v for _, v, _, _ in items) * 1.2)
    parts = [svg_open(w, h)]
    for i in range(nt + 1):
        gy = top + plot_h - plot_h * i * ystep / ymax
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{w - right}" '
                     f'y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(txt(left - 8, gy + 4, val_fmt(i * ystep), 11, MUTED,
                         "end", extra='style="font-variant-numeric:tabular-nums"'))
    n = len(items)
    band = plot_w / n
    bw = min(44, band * 0.5)
    for i, (lab, v, color, tip) in enumerate(items):
        x = left + band * i + (band - bw) / 2
        bh = plot_h * v / ymax
        y = top + plot_h - bh
        parts.append(rrect_top(x, y, bw, bh, 4, color, tip))
        parts.append(txt(x + bw / 2, y - 8, val_fmt(v), 12, INK, "middle", "600"))
        parts.append(txt(left + band * i + band / 2, h - 24, lab, 11.5, INK2,
                         "middle"))
        if unit:
            parts.append(txt(left + band * i + band / 2, h - 9, unit, 10.5,
                             MUTED, "middle"))
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{w - right}" '
                 f'y2="{top + plot_h}" stroke="{BASE}" stroke-width="1"/>')
    parts.append("</svg>")
    return "".join(parts)


def km_chart(series, colors, w=980, h=420, xlab="Tenure (months)", order=None,
             label_pad=8):
    left, right, top, bottom = 58, 190, 16, 52
    plot_w, plot_h = w - left - right, h - top - bottom
    X = R["km_times"]
    parts = [svg_open(w, h)]
    for i in range(5):
        gy = top + plot_h * i / 4
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" '
                     f'y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(txt(left - 8, gy + 4, f"{100 - 25 * i}%", 11, MUTED, "end"))
    for m in range(0, 73, 12):
        gx = left + plot_w * m / 72
        parts.append(txt(gx, h - 26, m, 11, MUTED, "middle"))
    parts.append(txt(left + plot_w / 2, h - 8, xlab, 11.5, MUTED, "middle"))
    names = order or list(series.keys())
    ends = []
    for name in names:
        Y = series[name]
        pts = " ".join(f"{left + plot_w * t / 72:.1f},"
                       f"{top + plot_h * (1 - y):.1f}" for t, y in zip(X, Y))
        parts.append(f'<polyline points="{pts}" fill="none" '
                     f'stroke="{colors[name]}" stroke-width="2.4" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        # hover dots each 12 months
        for m in range(0, 73, 12):
            y = Y[m]
            cx, cy = left + plot_w * m / 72, top + plot_h * (1 - y)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" '
                         f'fill="transparent" data-tip="{esc(name)} · month {m} · '
                         f'{y*100:.1f}% retained"/>')
        ex, ey = left + plot_w, top + plot_h * (1 - Y[-1])
        ends.append((name, ey))
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4.5" '
                     f'fill="{colors[name]}" stroke="{SURFACE}" stroke-width="2"/>')
    # collision-avoided end labels
    ends.sort(key=lambda t: t[1])
    placed = []
    for name, ey in ends:
        y = ey
        for py in placed:
            if abs(y - py) < 16:
                y = py + 16
        placed.append(y)
        parts.append(txt(left + plot_w + label_pad + 4, y + 4,
                         f"{name} · {series[name][-1]*100:.0f}%", 12, INK2))
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" '
                 f'x2="{left + plot_w}" y2="{top + plot_h}" stroke="{BASE}" '
                 f'stroke-width="1"/>')
    parts.append("</svg>")
    return "".join(parts)


def legend(pairs):
    row = "".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>'
        f'{esc(n)}</span>' for n, c in pairs)
    return f'<div class="legend">{row}</div>'


def table(headers, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in row) +
                  "</tr>" for row in rows)
    return (f'<details class="tbl"><summary>View data</summary>'
            f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'
            f'</details>')


def card(title, sub, body, span=6):
    return (f'<section class="card s{span}"><header><h2>{esc(title)}</h2>'
            f'<p>{esc(sub)}</p></header>{body}</section>')


def kpi(label, value, note=""):
    n = f'<span class="note">{esc(note)}</span>' if note else ""
    return (f'<div class="kpi"><span class="klabel">{esc(label)}</span>'
            f'<span class="kvalue">{esc(value)}</span>{n}</div>')


NAV = [("01_executive_overview.html", "Overview"),
       ("02_customer_flow_sankey.html", "Customer flow"),
       ("03_segments.html", "Segments"),
       ("04_retention_survival.html", "Retention"),
       ("05_model_performance.html", "Models"),
       ("06_impact_planner.html", "Impact")]

CSS = """
:root{color-scheme:light}
*{margin:0;box-sizing:border-box}
body{background:%(PAGE)s;color:%(INK)s;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;padding:26px 30px 48px}
.wrap{max-width:1280px;margin:0 auto}
.masthead{display:flex;align-items:baseline;justify-content:space-between;
  flex-wrap:wrap;gap:10px;margin-bottom:6px}
.masthead h1{font-size:21px;font-weight:650;letter-spacing:-0.2px}
.masthead .src{font-size:12px;color:%(MUTED)s}
nav{display:flex;gap:4px;flex-wrap:wrap;margin:14px 0 22px;
  border-bottom:1px solid %(GRID)s;padding-bottom:12px}
nav a{font-size:12.5px;color:%(INK2)s;text-decoration:none;padding:6px 12px;
  border-radius:999px}
nav a:hover{background:#ecebe5}
nav a.on{background:%(NAVY)s;color:#fff}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:14px;margin-bottom:22px}
.kpi{background:%(SURFACE)s;border:1px solid rgba(43,42,40,.08);
  border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:3px}
.klabel{font-size:11.5px;color:%(MUTED)s}
.kvalue{font-size:27px;font-weight:650;letter-spacing:-0.3px}
.note{font-size:11px;color:%(INK2)s}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
.card{background:%(SURFACE)s;border:1px solid rgba(43,42,40,.08);
  border-radius:12px;padding:18px 20px 14px;overflow:hidden}
.card header{margin-bottom:12px}
.card h2{font-size:14px;font-weight:650}
.card header p{font-size:12px;color:%(MUTED)s;margin-top:2px}
.s12{grid-column:span 12}.s8{grid-column:span 8}.s7{grid-column:span 7}
.s6{grid-column:span 6}.s5{grid-column:span 5}.s4{grid-column:span 4}
@media(max-width:960px){.card{grid-column:span 12}}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:%(INK2)s;
  margin:2px 0 10px}
.lg{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block}
.tbl{margin-top:8px;border-top:1px solid %(GRID)s;padding-top:8px}
.tbl summary{font-size:11.5px;color:%(MUTED)s;cursor:pointer}
.tbl table{width:100%%;border-collapse:collapse;margin-top:8px;font-size:12px}
.tbl th{text-align:left;color:%(MUTED)s;font-weight:500;padding:4px 8px;
  border-bottom:1px solid %(GRID)s}
.tbl td{padding:4px 8px;border-bottom:1px solid %(GRID)s;color:%(INK2)s;
  font-variant-numeric:tabular-nums}
.foot{margin-top:26px;font-size:11.5px;color:%(MUTED)s;line-height:1.6}
#tip{position:fixed;pointer-events:none;background:%(INK)s;color:#fff;
  font-size:12px;padding:6px 10px;border-radius:7px;opacity:0;
  transition:opacity .08s;z-index:10;max-width:260px}
svg [data-tip]{cursor:default}
svg [data-tip]:hover{opacity:.85}
""" % dict(PAGE=PAGE, SURFACE=SURFACE, INK=INK, INK2=INK2, MUTED=MUTED,
           GRID=GRID, NAVY=NAVY)

JS = """
const tip=document.createElement('div');tip.id='tip';document.body.appendChild(tip);
document.addEventListener('mousemove',e=>{
  const t=e.target.closest('[data-tip]');
  if(t){tip.textContent=t.dataset.tip;tip.style.opacity=1;
    tip.style.left=Math.min(e.clientX+14,innerWidth-280)+'px';
    tip.style.top=(e.clientY+16)+'px';}
  else tip.style.opacity=0;});
"""


def page(fname, title, sub, kpis_html, grid_html, foot_extra=""):
    nav = "".join(
        f'<a href="{f}" class="{"on" if f == fname else ""}">{n}</a>'
        for f, n in NAV)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' rx='4' fill='%233a6fa5'/></svg>">
<title>{esc(title)} — Customer Churn Analysis</title>
<style>{CSS}</style></head><body><div class="wrap">
<div class="masthead"><h1>{esc(title)}</h1>
<span class="src">Telco Customer Churn · 7,043 customers · regenerated from the analysis pipeline</span></div>
<nav>{nav}</nav>
{kpis_html}
<div class="grid">{grid_html}</div>
<p class="foot">{esc(sub)} {foot_extra}</p>
</div><script>{JS}</script></body></html>"""
    with open(os.path.join(OUT, fname), "w") as f:
        f.write(doc)
    print("wrote", fname, f"{os.path.getsize(os.path.join(OUT, fname))/1024:.0f}KB")


# ======================================================================
# 01 — EXECUTIVE OVERVIEW
# ======================================================================
kpis = "".join([
    kpi("Customers", fmt_n(R["n_customers"])),
    kpi("Churned", fmt_n(R["n_churned"]), f'{pct(R["churn_rate"])} of base'),
    kpi("Monthly recurring charges", money_k(R["monthly_revenue"])),
    kpi("Monthly charges lost to churn", money_k(R["monthly_revenue_at_risk"]),
        f'{pct(R["monthly_revenue_at_risk"]/R["monthly_revenue"])} of MRR'),
    kpi("Median tenure at churn", f'{R["median_tenure_churned"]:.0f} mo',
        f'retained: {R["median_tenure_retained"]:.0f} mo'),
])
kpis = f'<div class="kpis">{kpis}</div>'

cb = R["churn_by_Contract"]
c1 = column_chart(
    [("Month-to-month", cb["Month-to-month"]["rate"], SALMON,
      f'Month-to-month · {fmt_n(cb["Month-to-month"]["n"])} customers · '
      f'{pct(cb["Month-to-month"]["rate"])} churn'),
     ("One-year", cb["One year"]["rate"], GRAY,
      f'One-year · {fmt_n(cb["One year"]["n"])} customers · '
      f'{pct(cb["One year"]["rate"])} churn'),
     ("Two-year", cb["Two year"]["rate"], GRAY,
      f'Two-year · {fmt_n(cb["Two year"]["n"])} customers · '
      f'{pct(cb["Two year"]["rate"])} churn')],
    h=290, val_fmt=lambda v: f"{v*100:.0f}%")
tb1 = table(["Contract", "Customers", "Churned", "Churn rate"],
            [(k, fmt_n(v["n"]), fmt_n(v["churned"]), pct(v["rate"]))
             for k, v in cb.items()])

tbands = R["churn_by_tenure_band"]
c2 = column_chart(
    [(k, v["rate"], SALMON if v["rate"] > R["churn_rate"] else GRAY,
      f'{k} · {fmt_n(v["n"])} customers · {pct(v["rate"])} churn')
     for k, v in tbands.items()],
    h=290, val_fmt=lambda v: f"{v*100:.0f}%")
tb2 = table(["Tenure cohort", "Customers", "Churned", "Churn rate"],
            [(k, fmt_n(v["n"]), fmt_n(v["churned"]), pct(v["rate"]))
             for k, v in tbands.items()])

drivers = [
    ("Electronic check", R["churn_by_PaymentMethod"]["Electronic check"]["rate"]),
    ("Month-to-month", cb["Month-to-month"]["rate"]),
    ("Fiber-optic internet", R["churn_by_InternetService"]["Fiber optic"]["rate"]),
    ("Senior citizen", R["churn_by_SeniorCitizen"]["Yes"]["rate"]),
    ("No tech support", R["churn_by_TechSupport"]["No"]["rate"]),
    ("Paperless billing", R["churn_by_PaperlessBilling"]["Yes"]["rate"]),
    ("No online security", R["churn_by_OnlineSecurity"]["No"]["rate"]),
]
drivers.sort(key=lambda t: -t[1])
c3 = barh_chart([(n, v, SALMON, f"{n} · {pct(v)} churn rate") for n, v in drivers],
                w=620, xmax=0.55, ref=R["churn_rate"],
                ref_label=f'overall {pct(R["churn_rate"])}',
                val_fmt=lambda v: f"{v*100:.0f}%")
tb3 = table(["Attribute", "Churn rate"], [(n, pct(v)) for n, v in drivers])

segs = R["segments"]
c4 = barh_chart(
    [(n, segs[n]["monthly_revenue_churned"], SEG_COLORS[n],
      f'{n} · {money_k(segs[n]["monthly_revenue_churned"])}/mo lost · '
      f'{pct(segs[n]["churn_rate"])} churn') for n in SEG_ORDER],
    w=620, val_fmt=lambda v: money_k(v),
    suffix_fmt=lambda v, i: money_k(v) + "/mo")
tb4 = table(["Segment", "Customers", "Churn rate", "Monthly revenue lost"],
            [(n, fmt_n(segs[n]["n"]), pct(segs[n]["churn_rate"]),
              money_k(segs[n]["monthly_revenue_churned"])) for n in SEG_ORDER])

grid = "".join([
    card("Churn rate by contract type",
         "Month-to-month contracts churn at 15× the two-year rate", c1 + tb1),
    card("Churn rate by tenure cohort",
         "Nearly half of first-year customers churn", c2 + tb2),
    card("Churn rate by customer attribute",
         "Attributes ranked against the 26.5% base rate", c3 + tb3),
    card("Monthly revenue lost by segment",
         "New high-spend customers drive 66% of churned revenue", c4 + tb4),
])
page("01_executive_overview.html", "Executive overview",
     "All values computed from the checked-in dataset "
     "(Analysis/data/telco_churn_cleaned.csv) with seed 42.",
     kpis, grid)

# ======================================================================
# 02 — CUSTOMER FLOW SANKEY
# ======================================================================
def sankey_svg(w=1180, h=560):
    left_x, mid_x, right_x = 40, w / 2 - 60, w - 250
    node_w = 14
    top, bottom = 46, 26
    plot_h = h - top - bottom
    total = R["n_customers"]
    gapv = 26

    contracts = ["Month-to-month", "One year", "Two year"]
    bands = ["0-12 mo", "13-48 mo", "49-72 mo"]
    outcomes = ["Retained", "Churned"]
    flows1 = {(f["src"], f["dst"]): f["n"] for f in R["sankey"]
              if f["src"] in contracts}
    flows2 = {(f["src"], f["dst"]): f["n"] for f in R["sankey"]
              if f["src"] in bands}

    def totals(names, axis_flows, key):
        return {n: sum(v for k, v in axis_flows.items() if k[key] == n)
                for n in names}

    t_c = totals(contracts, flows1, 0)
    t_b = totals(bands, flows1, 1)
    t_o = totals(outcomes, flows2, 1)
    scale = (plot_h - gapv * 2) / total

    def stack(names, tot, x):
        pos, y = {}, top
        for n in names:
            hh = tot[n] * scale
            pos[n] = [x, y, hh, y]  # x, y0, height, cursor
            y += hh + gapv
        return pos

    P_c = stack(contracts, t_c, left_x)
    P_b = stack(bands, t_b, mid_x)
    P_o = stack(outcomes, t_o, right_x)
    # separate cursors for band inflow vs outflow
    b_in = {n: P_b[n][1] for n in bands}
    b_out = {n: P_b[n][1] for n in bands}

    parts = [svg_open(w, h)]

    def ribbon(x0, y0, x1, y1, hh, color, opacity, tip):
        c = (x1 - x0) * 0.45
        d = (f"M{x0:.1f},{y0:.1f} C{x0 + c:.1f},{y0:.1f} {x1 - c:.1f},{y1:.1f} "
             f"{x1:.1f},{y1:.1f} v{hh:.1f} C{x1 - c:.1f},{y1 + hh:.1f} "
             f"{x0 + c:.1f},{y0 + hh:.1f} {x0:.1f},{y0 + hh:.1f} z")
        return (f'<path d="{d}" fill="{color}" opacity="{opacity}" '
                f'data-tip="{esc(tip)}"/>')

    for cn in contracts:
        for bn in bands:
            n = flows1.get((cn, bn), 0)
            if not n:
                continue
            hh = n * scale
            y0 = P_c[cn][3]; P_c[cn][3] += hh
            y1 = b_in[bn]; b_in[bn] += hh
            parts.append(ribbon(left_x + node_w, y0, mid_x, y1, hh, GRAY, 0.38,
                                f"{cn} → {bn} · {fmt_n(n)} customers"))
    for bn in bands:
        for on in outcomes:
            n = flows2.get((bn, on), 0)
            if not n:
                continue
            hh = n * scale
            y0 = b_out[bn]; b_out[bn] += hh
            y1 = P_o[on][3]; P_o[on][3] += hh
            color = SALMON if on == "Churned" else BLUE
            parts.append(ribbon(mid_x + node_w, y0, right_x, y1, hh, color,
                                0.42, f"{bn} → {on} · {fmt_n(n)} customers"))

    halo = (f'style="paint-order:stroke;stroke:{SURFACE};stroke-width:4px;'
            f'stroke-linejoin:round"')

    def node(pos, names, tot, colors, side="right"):
        for n in names:
            x, y, hh, _ = pos[n]
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{node_w}" '
                         f'height="{hh:.1f}" rx="3" fill="{colors(n)}" '
                         f'data-tip="{esc(n)} · {fmt_n(tot[n])} customers"/>')
            share = tot[n] / total
            if side == "right":
                parts.append(txt(x + node_w + 9, y + hh / 2 - 2, n, 12.5, INK,
                                 weight="600", extra=halo))
                parts.append(txt(x + node_w + 9, y + hh / 2 + 13,
                                 f"{fmt_n(tot[n])} · {pct(share)}", 11, INK2,
                                 extra=halo))
            else:
                parts.append(txt(x - 9, y + hh / 2 - 2, n, 12.5, INK, "end",
                                 "600", extra=halo))
                parts.append(txt(x - 9, y + hh / 2 + 13,
                                 f"{fmt_n(tot[n])} · {pct(share)}", 11, INK2,
                                 "end", extra=halo))

    node(P_c, contracts, t_c, lambda n: NAVY)
    node(P_b, bands, t_b, lambda n: GRAY)
    node(P_o, outcomes, t_o,
         lambda n: SALMON if n == "Churned" else BLUE, side="left")

    for label, x in [("Contract", left_x), ("Tenure", mid_x),
                     ("Outcome", right_x)]:
        parts.append(txt(x + node_w / 2, 24, label.upper(), 11, MUTED, "middle",
                         extra='letter-spacing="1.2"'))
    parts.append("</svg>")
    return "".join(parts)


flow_rows = ([(f["src"], f["dst"], fmt_n(f["n"])) for f in R["sankey"]])
sk_table = table(["From", "To", "Customers"], flow_rows)

inet = R["churn_by_InternetService"]
c_inet = column_chart(
    [("Fiber optic", inet["Fiber optic"]["rate"], SALMON,
      f'Fiber optic · {fmt_n(inet["Fiber optic"]["n"])} customers · '
      f'{pct(inet["Fiber optic"]["rate"])} churn'),
     ("DSL", inet["DSL"]["rate"], GRAY,
      f'DSL · {fmt_n(inet["DSL"]["n"])} customers · {pct(inet["DSL"]["rate"])} churn'),
     ("No internet", inet["No"]["rate"], GRAY,
      f'No internet · {fmt_n(inet["No"]["n"])} customers · {pct(inet["No"]["rate"])} churn')],
    h=280, val_fmt=lambda v: f"{v*100:.0f}%")

pm = R["churn_by_PaymentMethod"]
pm_items = sorted(pm.items(), key=lambda kv: -kv[1]["rate"])
c_pm = barh_chart(
    [(k, v["rate"], SALMON if v["rate"] > R["churn_rate"] else GRAY,
      f'{k} · {fmt_n(v["n"])} customers · {pct(v["rate"])} churn')
     for k, v in pm_items],
    w=620, xmax=0.55, ref=R["churn_rate"],
    ref_label=f'overall {pct(R["churn_rate"])}',
    val_fmt=lambda v: f"{v*100:.0f}%")

grid = "".join([
    card("Customer flow — contract to tenure to outcome",
         "Ribbon width is customer count; churn concentrates in "
         "month-to-month contracts that never clear the first year",
         sankey_svg() + sk_table, span=12),
    card("Churn rate by internet service",
         "Fiber-optic customers churn at twice the DSL rate", c_inet +
         table(["Internet service", "Customers", "Churn rate"],
               [(k, fmt_n(v["n"]), pct(v["rate"])) for k, v in inet.items()])),
    card("Churn rate by payment method",
         "Electronic check stands apart from automatic payment methods", c_pm +
         table(["Payment method", "Customers", "Churn rate"],
               [(k, fmt_n(v["n"]), pct(v["rate"])) for k, v in pm_items])),
])
page("02_customer_flow_sankey.html", "Customer flow",
     "Flows computed from contract type, tenure band, and churn outcome of "
     "all 7,043 customers.", "", grid)

# ======================================================================
# 03 — SEGMENTS
# ======================================================================
kpis = "".join(
    kpi(n, pct(segs[n]["churn_rate"]),
        f'{fmt_n(segs[n]["n"])} customers · ${segs[n]["avg_monthly"]:.0f}/mo avg')
    for n in SEG_ORDER)
kpis = f'<div class="kpis">{kpis}</div>'


def scatter_svg(w=1180, h=470):
    left, right, top, bottom = 64, 220, 34, 54
    pw, ph = w - left - right, h - top - bottom
    parts = [svg_open(w, h)]
    for i in range(5):
        gy = top + ph - ph * i / 4
        val = 30 * i
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" '
                     f'y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(txt(left - 8, gy + 4, f"${val}", 11, MUTED, "end"))
    for m in range(0, 73, 12):
        parts.append(txt(left + pw * m / 72, h - 30, m, 11, MUTED, "middle"))
    parts.append(txt(left + pw / 2, h - 10, "Tenure (months)", 11.5, MUTED,
                     "middle"))
    parts.append(txt(18, 14, "Monthly charges", 11.5, MUTED))
    focus = "New · High spend"
    pts = sorted(R["segment_scatter"], key=lambda p: p["seg"] == focus)
    for p in pts:
        x = left + pw * p["t"] / 72
        y = top + ph * (1 - p["m"] / 120)
        hot = p["seg"] == focus
        color = SALMON if hot else GRAY
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4.4 if hot else 3.4}" '
            f'fill="{color}" opacity="{0.85 if hot else 0.45}" '
            f'stroke="{SURFACE}" stroke-width="1" '
            f'data-tip="{esc(p["seg"])} · {p["t"]} mo · ${p["m"]:.0f}/mo · '
            f'{"churned" if p["churn"] else "retained"}"/>')
    parts.append(txt(left + pw + 16, top + 30, "New · High spend", 12.5, INK,
                     weight="600"))
    parts.append(txt(left + pw + 16, top + 48, "48.2% churn — short tenure,", 11.5, INK2))
    parts.append(txt(left + pw + 16, top + 64, "high monthly charges", 11.5, INK2))
    parts.append(txt(left + pw + 16, top + 92, "All other segments in gray", 11.5, MUTED))
    parts.append("</svg>")
    return "".join(parts)


c_rates = barh_chart(
    [(n, segs[n]["churn_rate"], SEG_COLORS[n],
      f'{n} · {pct(segs[n]["churn_rate"])} churn · '
      f'{fmt_n(segs[n]["n"])} customers') for n in SEG_ORDER],
    w=620, xmax=0.6, ref=R["churn_rate"],
    ref_label=f'overall {pct(R["churn_rate"])}',
    val_fmt=lambda v: f"{v*100:.0f}%", suffix_fmt=lambda v, i: pct(v))

mix = barh_chart(
    [(n, segs[n]["mtm_share"], SEG_COLORS[n],
      f'{n} · {pct(segs[n]["mtm_share"])} on month-to-month contracts')
     for n in SEG_ORDER],
    w=620, xmax=1.0, val_fmt=lambda v: f"{v*100:.0f}%")

seg_tbl = table(
    ["Segment", "Customers", "Churn rate", "Avg tenure", "Avg monthly",
     "Month-to-month", "Fiber share", "Monthly revenue lost"],
    [(n, fmt_n(segs[n]["n"]), pct(segs[n]["churn_rate"]),
      f'{segs[n]["avg_tenure"]:.0f} mo', f'${segs[n]["avg_monthly"]:.0f}',
      pct(segs[n]["mtm_share"]), pct(segs[n]["fiber_share"]),
      money_k(segs[n]["monthly_revenue_churned"])) for n in SEG_ORDER])

grid = "".join([
    card("Customer landscape — tenure × monthly charges",
         "K-Means (k=4) on tenure, monthly and total charges; the high-risk "
         "segment highlighted, 700-customer sample",
         scatter_svg() + seg_tbl, span=12),
    card("Churn rate by segment",
         "New high-spend customers churn at nearly half", c_rates),
    card("Month-to-month share by segment",
         "Contract mix explains most of the churn-rate gap", mix),
])
page("03_segments.html", "Customer segments",
     "Segments fit with scikit-learn KMeans (seed 42) on standardized tenure, "
     "monthly charges, and total charges.", kpis, grid)

# ======================================================================
# 04 — RETENTION / SURVIVAL
# ======================================================================
CONTRACT_COLORS = {"Two year": BLUE, "One year": VIOLET,
                   "Month-to-month": SALMON}
km_c = {k: v for k, v in R["km_contract"].items() if k != "All customers"}
km1 = km_chart(km_c, CONTRACT_COLORS, w=1180, h=460,
               order=["Two year", "One year", "Month-to-month"])
km1_tbl = table(
    ["Contract", "Retained @ 12 mo", "@ 24 mo", "@ 48 mo", "@ 72 mo"],
    [(k, pct(v[12]), pct(v[24]), pct(v[48]), pct(v[72]))
     for k, v in km_c.items()])

km2 = km_chart(R["km_segments"], SEG_COLORS, w=580, h=400, order=SEG_ORDER,
               label_pad=2)

hr = R["cox_hazard_ratios"]
hr_names = {
    "InternetService_Fiber optic": "Fiber-optic internet",
    "PaymentMethod_Electronic check": "Electronic check",
    "PaymentMethod_Mailed check": "Mailed check",
    "PaperlessBilling_Yes": "Paperless billing",
    "SeniorCitizen_Yes": "Senior citizen",
    "Partner_Yes": "Has partner",
    "OnlineSecurity_Yes": "Online security",
    "TechSupport_Yes": "Tech support",
    "Contract_One year": "One-year contract",
    "Contract_Two year": "Two-year contract",
}


def forest_svg(w=580, h=400):
    picks = ["InternetService_Fiber optic", "PaymentMethod_Mailed check",
             "PaymentMethod_Electronic check", "PaperlessBilling_Yes",
             "SeniorCitizen_Yes", "Partner_Yes", "OnlineSecurity_Yes",
             "TechSupport_Yes", "Contract_One year", "Contract_Two year"]
    picks = [p for p in picks if p in hr]
    picks.sort(key=lambda p: -hr[p]["hr"])
    left, right, top, bottom = 168, 84, 14, 40
    pw = w - left - right
    row_h = (h - top - bottom) / len(picks)
    lo, hi = np.log(0.05), np.log(3.2)

    def X(v):
        return left + pw * (np.log(v) - lo) / (hi - lo)

    parts = [svg_open(w, h)]
    for v in [0.1, 0.25, 0.5, 1, 2]:
        gx = X(v)
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" '
                     f'y2="{h - bottom + 4}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(txt(gx, h - 22, f"{v:g}×", 11, MUTED, "middle"))
    gx = X(1)
    parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" '
                 f'y2="{h - bottom + 4}" stroke="{BASE}" stroke-width="1.4"/>')
    parts.append(txt(left + pw / 2, h - 6, "Hazard ratio (log scale)", 11,
                     MUTED, "middle"))
    for i, p in enumerate(picks):
        v = hr[p]
        y = top + row_h * i + row_h / 2
        color = SALMON if v["hr"] > 1 else BLUE
        parts.append(txt(left - 10, y + 4, hr_names.get(p, p), 12, INK2, "end"))
        parts.append(f'<line x1="{X(max(v["lo"], 0.05)):.1f}" y1="{y:.1f}" '
                     f'x2="{X(min(v["hi"], 3.2)):.1f}" y2="{y:.1f}" '
                     f'stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{X(v["hr"]):.1f}" cy="{y:.1f}" r="5" '
                     f'fill="{color}" stroke="{SURFACE}" stroke-width="1.6" '
                     f'data-tip="{esc(hr_names.get(p, p))} · HR {v["hr"]:.2f} '
                     f'[{v["lo"]:.2f}–{v["hi"]:.2f}]"/>')
        parts.append(txt(w - right + 10, y + 4, f'{v["hr"]:.2f}', 12, INK,
                         weight="600"))
    parts.append("</svg>")
    return "".join(parts)


ab = sorted(R["ab_tests"], key=lambda s: -s["relative_lift"])
ab_names = {"Discount Offer": "Discount offer",
            "Free Premium Technical Support": "Free tech support",
            "Free Service Upgrade": "Service upgrade",
            "Contract Incentive": "Contract incentive",
            "Enhanced Loyalty Program": "Loyalty program"}
best_roi = max(s["estimated_roi"] for s in ab)
c_ab = barh_chart(
    [(ab_names[s["strategy_name"]], s["relative_lift"] / 100,
      GOLD if s["estimated_roi"] == best_roi else GRAY,
      f'{ab_names[s["strategy_name"]]} · target: {s["target_segment"]} · '
      f'+{s["relative_lift"]:.1f}% lift · ROI {s["estimated_roi"]:.0f}% · '
      f'p={s["p_value"]:.2g}') for s in ab],
    w=1180, xmax=0.3, left=190,
    val_fmt=lambda v: f"{v*100:.0f}%",
    suffix_fmt=lambda v, i: f'+{v*100:.1f}% · ROI {ab[i]["estimated_roi"]:.0f}%')
ab_tbl = table(
    ["Strategy", "Target segment", "Retention lift", "p-value", "Cost/customer",
     "Simulated ROI"],
    [(ab_names[s["strategy_name"]], s["target_segment"],
      f'+{s["relative_lift"]:.1f}%', f'{s["p_value"]:.2g}',
      f'${s["cost_per_customer"]}', f'{s["estimated_roi"]:.0f}%') for s in ab])

kpis = "".join([
    kpi("2-yr contract retention @ 72 mo", pct(km_c["Two year"][72])),
    kpi("Month-to-month @ 72 mo", pct(km_c["Month-to-month"][72])),
    kpi("Strongest risk factor", "Fiber optic",
        f'Cox HR {hr["InternetService_Fiber optic"]["hr"]:.2f}'),
    kpi("Strongest protective factor", "Two-year contract",
        f'Cox HR {hr["Contract_Two year"]["hr"]:.2f}'),
    kpi("Best intervention ROI", "Free tech support", "+22.8% lift · ROI 160%"),
])
kpis = f'<div class="kpis">{kpis}</div>'

grid = "".join([
    card("Kaplan-Meier retention by contract type",
         "Hover any curve for month-by-month retention",
         legend([(n, CONTRACT_COLORS[n]) for n in
                 ["Two year", "One year", "Month-to-month"]]) + km1 + km1_tbl,
         span=12),
    card("Retention by customer segment",
         "K-Means segments; new high-spend customers collapse fastest",
         legend([(n, SEG_COLORS[n]) for n in SEG_ORDER]) + km2),
    card("Cox proportional-hazards model",
         "Hazard ratios with 95% CI; above 1× raises churn risk",
         forest_svg()),
    card("Recommended interventions — simulated A/B tests",
         "Five strategies evaluated with power analysis; gold = best ROI",
         c_ab + ab_tbl, span=12),
])
page("04_retention_survival.html", "Retention & survival",
     "Kaplan-Meier and Cox models fit with lifelines on all 7,043 customers; "
     "A/B results are seeded simulations.", kpis, grid)

# ======================================================================
# 05 — MODEL PERFORMANCE
# ======================================================================
bench = R["benchmark"]
winner = R["winner"]
order = sorted(bench, key=lambda k: -bench[k]["test_accuracy"])

kpis = "".join([
    kpi("Best holdout accuracy", pct(bench[winner]["test_accuracy"]), winner),
    kpi("Best ROC AUC", f'{max(b["roc_auc"] for b in bench.values()):.3f}',
        max(bench, key=lambda k: bench[k]["roc_auc"])),
    kpi("Models benchmarked", str(len(bench)), "5-fold CV + 20% holdout"),
    kpi("Holdout size", fmt_n(R["test_size"]), "stratified, seed 42"),
])
kpis = f'<div class="kpis">{kpis}</div>'

c_bench = barh_chart(
    [(k, bench[k]["test_accuracy"], GOLD if k == winner else GRAY,
      f'{k} · accuracy {pct(bench[k]["test_accuracy"])} · '
      f'AUC {bench[k]["roc_auc"]:.3f} · F1 {bench[k]["f1"]:.3f}')
     for k in order],
    w=1180, xmax=0.9, left=190, bar_h=18, gap=12,
    val_fmt=lambda v: f"{v*100:.0f}%", suffix_fmt=lambda v, i: pct(v))
bench_tbl = table(
    ["Model", "CV accuracy", "Holdout accuracy", "ROC AUC", "Precision",
     "Recall", "F1"],
    [(k, f'{bench[k]["cv_accuracy_mean"]:.3f} ± {bench[k]["cv_accuracy_std"]:.3f}',
      pct(bench[k]["test_accuracy"]), f'{bench[k]["roc_auc"]:.3f}',
      f'{bench[k]["precision"]:.3f}', f'{bench[k]["recall"]:.3f}',
      f'{bench[k]["f1"]:.3f}') for k in order])


def roc_svg(w=580, h=430):
    left, right, top, bottom = 56, 20, 14, 52
    pw, ph = w - left - right, h - top - bottom
    parts = [svg_open(w, h)]
    for i in range(5):
        gy = top + ph * i / 4
        gx = left + pw * i / 4
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" '
                     f'y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(txt(left - 8, gy + 4, f"{1 - i/4:.2g}", 11, MUTED, "end"))
        parts.append(txt(gx, h - 30, f"{i/4:.2g}", 11, MUTED, "middle"))
    parts.append(f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" '
                 f'y2="{top}" stroke="{BASE}" stroke-width="1" '
                 f'stroke-dasharray="1 4" stroke-linecap="round"/>')
    parts.append(txt(left + pw / 2, h - 10, "False-positive rate", 11.5, MUTED,
                     "middle"))
    parts.append(txt(16, top + 2, "TPR", 11, MUTED))

    def line(name, color, width, opacity=1.0):
        rc = R["roc_curves"][name]
        pts = " ".join(f"{left + pw * f:.1f},{top + ph * (1 - t):.1f}"
                       for f, t in zip(rc["fpr"], rc["tpr"]))
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" opacity="{opacity}" '
                f'stroke-linecap="round" data-tip="{esc(name)} · AUC '
                f'{bench[name]["roc_auc"]:.3f}"/>')

    context_models = [n for n in ["Random Forest", "Gradient Boosting",
                                  "AdaBoost", "KNN", "SVM (RBF)",
                                  "LightGBM (Optuna)", "XGBoost (Optuna)",
                                  "Stacked Ensemble v2"]
                      if n in R["roc_curves"] and n != winner]
    for name in context_models:
        parts.append(line(name, GRAY, 1.4, 0.7))
    parts.append(line(winner, SALMON, 2.6))
    parts.append(txt(left + pw * 0.34, top + ph * 0.22, winner, 12.5, INK,
                     weight="600"))
    parts.append(txt(left + pw * 0.34, top + ph * 0.22 + 16,
                     f'AUC {bench[winner]["roc_auc"]:.3f}', 11.5, INK2))
    parts.append(txt(left + pw * 0.62, top + ph * 0.52, "other models", 11.5,
                     MUTED))
    parts.append("</svg>")
    return "".join(parts)


def confusion_svg(w=580, h=430):
    cm = np.array(R["winner_confusion"])
    total = cm.sum()
    left, top = 120, 60
    cell = 150
    labels = [["True negative", "False positive"],
              ["False negative", "True positive"]]
    shades = ["#16436e", "#d5e0ee", "#c3d2e6", "#9fb8d6"]
    fills = [[shades[0], shades[1]], [shades[2], shades[3]]]
    parts = [svg_open(w, h)]
    parts.append(txt(left + cell, 26, "Predicted", 11, MUTED, "middle",
                     extra='letter-spacing="1"'))
    for j, lab in enumerate(["Stay", "Churn"]):
        parts.append(txt(left + cell * j + cell / 2, 46, lab, 12, INK2,
                         "middle"))
    for i, lab in enumerate(["Stayed", "Churned"]):
        parts.append(txt(left - 14, top + cell * i + cell / 2 + 4, lab, 12,
                         INK2, "end"))
    for i in range(2):
        for j in range(2):
            n = int(cm[i, j])
            share = n / total
            fill = fills[i][j]
            dark = fill == shades[0]
            fg = "#ffffff" if dark else INK
            fg2 = "#cfdcea" if dark else INK2
            x, y = left + cell * j, top + cell * i
            parts.append(
                f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" '
                f'height="{cell - 4}" rx="10" fill="{fill}" '
                f'data-tip="{labels[i][j]} · {fmt_n(n)} customers · '
                f'{share*100:.1f}% of holdout"/>')
            parts.append(txt(x + cell / 2, y + cell / 2 - 2, fmt_n(n), 27, fg,
                             "middle", "650"))
            parts.append(txt(x + cell / 2, y + cell / 2 + 22,
                             f"{labels[i][j]} · {share*100:.1f}%", 11, fg2,
                             "middle"))
    m = bench[winner]
    y0 = top + 2 * cell + 34
    parts.append(txt(left - 14 + 0, y0,
                     f'Precision {m["precision"]:.3f} · Recall {m["recall"]:.3f} · '
                     f'F1 {m["f1"]:.3f}', 12.5, INK2))
    parts.append("</svg>")
    return "".join(parts)


imp = R["feature_importance_gb"]
imp_names = {"tenure": "Tenure", "InternetService_Fiber optic": "Fiber-optic internet",
             "PaymentMethod_Electronic check": "Electronic check",
             "Contract_Two year": "Two-year contract",
             "TotalCharges": "Total charges", "Contract_One year": "One-year contract",
             "MonthlyCharges": "Monthly charges",
             "OnlineSecurity_Yes": "Online security",
             "TechSupport_Yes": "Tech support",
             "PaperlessBilling_Yes": "Paperless billing",
             "gender_Male": "Gender (male)",
             "StreamingTV_Yes": "Streaming TV",
             "MultipleLines_Yes": "Multiple lines",
             "OnlineBackup_Yes": "Online backup",
             "SeniorCitizen_Yes": "Senior citizen",
             "Partner_Yes": "Has partner",
             "Dependents_Yes": "Has dependents"}
imp_items = list(imp.items())[:9]
c_imp = barh_chart(
    [(imp_names.get(k, k), v, BLUE, f'{imp_names.get(k, k)} · importance {v:.3f}')
     for k, v in imp_items],
    w=1180, left=190, val_fmt=lambda v: f"{v:.2f}",
    suffix_fmt=lambda v, i: f"{v:.3f}")

thr_note = ""
if R.get("winner_threshold"):
    thr_note = (f"; {winner} scored at its CV-tuned threshold "
                f"({R['winner_threshold']:.2f}), others at 0.50")
grid = "".join([
    card("Classifier benchmark — holdout accuracy",
         f"{len(bench)} models, identical preprocessing and split; "
         f"gold = best accuracy{thr_note}",
         c_bench + bench_tbl, span=12),
    card("ROC curves — holdout set",
         "Winner emphasized against the field", roc_svg()),
    card(f"Confusion matrix — {winner.lower()}",
         f"1,409 holdout customers at the "
         f"{R.get('winner_threshold', 0.5):.2f} threshold", confusion_svg()),
    card("Feature importance — gradient boosting",
         "Tenure, service type, and payment method dominate", c_imp, span=12),
])
page("05_model_performance.html", "Model performance",
     "Benchmark on an 80/20 stratified split of the 7,043-customer dataset, "
     "5-fold cross-validation on the training split, seed 42.", kpis, grid)

# ======================================================================
# 06 — IMPACT PLANNER
# ======================================================================
if "impact" in R:
    imp_rows = R["impact"]
    tgt = [r for r in imp_rows if r["mode"] == "model-targeted"]
    bl = {r["strategy"]: r for r in imp_rows if r["mode"] == "blanket"}
    tgt.sort(key=lambda r: -r["net_12mo"])
    pos = [r for r in tgt if r["net_12mo"] > 0]
    lift = R["final_lift"]["top_30"]

    kpis = "".join([
        kpi("Monthly revenue lost to churn", money_k(R["monthly_revenue_at_risk"]),
            "addressable base"),
        kpi("Projected customers saved / yr",
            fmt_n(sum(r["customers_saved"] for r in pos)),
            "positive-ROI strategies, model-targeted"),
        kpi("Projected revenue protected",
            money_k(sum(r["monthly_revenue_saved"] for r in pos)) + "/mo",
            f'{money_k(sum(r["monthly_revenue_saved"] for r in pos) * 12)} over 12 mo'),
        kpi("Program cost", money_k(sum(r["cost"] for r in pos)),
            "one-time, top-30% risk targeting"),
        kpi("Net 12-month impact",
            money_k(sum(r["net_12mo"] for r in pos)),
            f'{sum(r["net_12mo"] for r in pos) / max(sum(r["cost"] for r in pos), 1) * 100:.0f}% ROI'),
    ])
    kpis = f'<div class="kpis">{kpis}</div>'

    strat_short = {"Discount Offer": "Discount offer",
                   "Free Premium Technical Support": "Free tech support",
                   "Free Service Upgrade": "Service upgrade",
                   "Contract Incentive": "Contract incentive",
                   "Enhanced Loyalty Program": "Loyalty program"}

    # net impact: model-targeted vs blanket (diverging barh, two series)
    def net_chart(w=1180):
        rows = tgt
        left, right = 320, 150
        bar_h, gap, gap2 = 16, 6, 26
        top, bottom = 24, 30
        n = len(rows)
        h = top + n * (bar_h * 2 + gap + gap2) - gap2 + bottom
        pw = w - left - right
        vals = [r["net_12mo"] for r in rows] + [bl[r["strategy"]]["net_12mo"]
                                                for r in rows]
        vmax, vmin = max(vals), min(vals)
        span = vmax - vmin
        x0 = left + pw * (0 - vmin) / span
        parts = [svg_open(w, h)]
        parts.append(f'<line x1="{x0:.1f}" y1="{top - 6}" x2="{x0:.1f}" '
                     f'y2="{h - bottom + 4}" stroke="{BASE}" stroke-width="1.3"/>')
        for i, r in enumerate(rows):
            y = top + i * (bar_h * 2 + gap + gap2)
            parts.append(txt(left - 10, y + bar_h + gap / 2 + 4,
                             strat_short[r["strategy"]], 12, INK2, "end"))
            for j, (rr, color) in enumerate([(r, BLUE),
                                             (bl[r["strategy"]], GRAY)]):
                v = rr["net_12mo"]
                yy = y + j * (bar_h + gap)
                bw = abs(pw * v / span)
                bx = x0 if v >= 0 else x0 - bw
                tip = (f'{strat_short[r["strategy"]]} · {rr["mode"]} · contact '
                       f'{fmt_n(rr["contacted"])} · save {rr["customers_saved"]:.0f} '
                       f'customers · net {money_k(rr["net_12mo"])} · '
                       f'ROI {rr["roi_pct"]:.0f}%')
                if v >= 0:
                    parts.append(rrect_right(bx, yy, bw, bar_h, 4, color, tip))
                else:
                    parts.append(f'<rect x="{bx:.1f}" y="{yy}" width="{bw:.1f}" '
                                 f'height="{bar_h}" rx="4" fill="{color}" '
                                 f'data-tip="{esc(tip)}"/>')
                lx = x0 + (bw + 8 if v >= 0 else 8)
                parts.append(txt(lx, yy + bar_h / 2 + 4, money_k(v), 11.5, INK,
                                 "start", "600"))
        parts.append(txt(x0, h - 8, "$0", 11, MUTED, "middle"))
        parts.append(txt(left + pw, h - 8, "net 12-month impact →", 11, MUTED,
                         "end"))
        parts.append("</svg>")
        return "".join(parts)

    net_tbl = table(
        ["Strategy", "Mode", "Contacted", "Customers saved", "Revenue saved /mo",
         "Cost", "Net (12 mo)", "ROI"],
        [(strat_short[r["strategy"]], r["mode"], fmt_n(r["contacted"]),
          f'{r["customers_saved"]:.0f}', money_k(r["monthly_revenue_saved"]),
          money_k(r["cost"]), money_k(r["net_12mo"]), f'{r["roi_pct"]:.0f}%')
         for r in sorted(imp_rows, key=lambda x: (x["strategy"], x["mode"]))])

    # what you might see: scenario bands (50% / 100% of simulated lift)
    def scenario_chart(w=1180):
        rows = tgt
        items = []
        for r in rows:
            items.append((strat_short[r["strategy"]],
                          r["monthly_revenue_saved"],
                          BLUE if r["net_12mo"] > 0 else GRAY,
                          f'{strat_short[r["strategy"]]} · expected '
                          f'{money_k(r["monthly_revenue_saved"])}/mo · conservative '
                          f'{money_k(r["monthly_revenue_saved"] * 0.5)}/mo'))
        return barh_chart(items, w=w, left=200,
                          val_fmt=lambda v: money_k(v),
                          suffix_fmt=lambda v, i: f'{money_k(v)}/mo expected · '
                                                  f'{money_k(v*0.5)}/mo conservative')

    grid = "".join([
        card("Net 12-month impact — model-targeted vs blanket outreach",
             "Model targeting contacts the top-30% risk scores "
             f'({lift["capture"]*100:.0f}% of churners reached at a third of the cost); '
             "blanket contacts the whole pool",
             legend([("Model-targeted", BLUE), ("Blanket", GRAY)]) +
             net_chart() + net_tbl, span=12),
        card("Projected monthly revenue protected by strategy",
             "Expected case uses the full simulated lift; conservative case "
             "assumes half the lift survives in production", scenario_chart(),
             span=12),
    ])
    page("06_impact_planner.html", "Impact planner",
         "Projections combine real cohort sizes and churner revenue with "
         "simulated A/B lift rates and final-model targeting capture. "
         "Save rate = (treatment − baseline retention) / (1 − baseline). "
         "12-month revenue horizon, one-time program cost.", kpis, grid)

print("done")
