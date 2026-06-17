/**
 * presence.js — client-side mirror of the backend presence-calendar maths.
 *
 * The presence calendar describes *when a building is occupied* as 12 month
 * patterns (full weeks, extra weekdays, weekends, visit probability) and the
 * home/away load model turns that into per-month min/max day counts. We
 * replicate that conversion here so the editor and the load-profile detail
 * page can show "~X giorni", the annual presence %, and the presence heatmap
 * live, without a server round-trip on every keystroke.
 *
 * Keep this in sync with
 * ``sim_stochastic_pv/simulation/load_profiles/presence.py``.
 */

// Days per calendar month (Jan→Dec), non-leap year — matches MONTH_LENGTHS.
export const MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
export const DAYS_PER_YEAR = 365;

// Annual energy of the unscaled ARERA baseline (kWh). Mirrors
// ARERA_BASELINE_ANNUAL_KWH in bolletta.py — keep in sync if BL_TABLE changes.
// Lets the bill-fit KPI (home/away split + scale) update live, client-side.
export const ARERA_BASELINE_ANNUAL_KWH = 1195.82;
export const MIN_HOME_SCALE_FACTOR = 0.05;

// Italian month names for editor rows / labels.
export const MONTH_NAMES = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre",
];
export const MONTH_ABBR = [
    "Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
    "Lug", "Ago", "Set", "Ott", "Nov", "Dic",
];

/** A single month's default pattern (home all weeks, no extras). */
export function emptyMonthPattern() {
    return { weekends: true, full_weeks: 4, extra_weekdays: 0, visit_probability: 0.0 };
}

/** Normalise a (possibly partial) month object, applying defaults. */
export function normalizeMonth(m) {
    const d = emptyMonthPattern();
    if (!m) return d;
    return {
        weekends: m.weekends === undefined ? d.weekends : !!m.weekends,
        full_weeks: clampInt(m.full_weeks ?? d.full_weeks, 0, 5),
        extra_weekdays: clampInt(m.extra_weekdays ?? d.extra_weekdays, 0, 7),
        visit_probability: clampNum(m.visit_probability ?? d.visit_probability, 0, 1),
    };
}

/** Ensure a calendar object has exactly 12 normalised months. */
export function normalizeCalendar(cal) {
    const months = (cal && Array.isArray(cal.months)) ? cal.months : [];
    const out = [];
    for (let i = 0; i < 12; i++) out.push(normalizeMonth(months[i]));
    return { months: out };
}

function clampInt(v, lo, hi) {
    const n = Math.round(Number(v));
    if (Number.isNaN(n)) return lo;
    return Math.max(lo, Math.min(hi, n));
}
function clampNum(v, lo, hi) {
    const n = Number(v);
    if (Number.isNaN(n)) return lo;
    return Math.max(lo, Math.min(hi, n));
}

/**
 * Min/max/expected occupied days for one month — mirrors
 * MonthPresencePattern.min_max_home_days + the (min+max)/2 expectation.
 */
export function monthMinMax(pattern, monthIndex) {
    const p = normalizeMonth(pattern);
    const L = MONTH_LENGTHS[monthIndex];
    const full = Math.min(p.full_weeks * 7, L);
    const leftover = L - full;
    const weekendDays = p.weekends ? Math.round((leftover * 2) / 7) : 0;
    let min = full + weekendDays + p.extra_weekdays;
    min = Math.max(0, Math.min(min, L));
    const remaining = L - min;
    const max = Math.min(min + Math.floor(remaining * p.visit_probability), L);
    return { min, max, expected: (min + max) / 2 };
}

/** Per-month {min[], max[], expected[]} arrays for a whole calendar. */
export function calendarMinMax(cal) {
    const { months } = normalizeCalendar(cal);
    const min = [], max = [], expected = [];
    months.forEach((m, i) => {
        const r = monthMinMax(m, i);
        min.push(r.min);
        max.push(r.max);
        expected.push(r.expected);
    });
    return { min, max, expected };
}

/** Annual occupancy fraction in [0,1] — sum(expected) / 365. */
export function annualPresenceFraction(cal) {
    const { expected } = calendarMinMax(cal);
    const frac = expected.reduce((a, b) => a + b, 0) / DAYS_PER_YEAR;
    return Math.max(0, Math.min(1, frac));
}

/** Total expected occupied days across the year (rounded). */
export function annualPresenceDays(cal) {
    const { expected } = calendarMinMax(cal);
    return expected.reduce((a, b) => a + b, 0);
}

// Presence-heatmap category indices (align with the Heatmap categorical mode).
export const PRESENCE_CATEGORIES = ["A casa (certo)", "Possibile visita", "Via"];
export const PRESENCE_CATEGORY_COLORS = ["#1d4ed8", "#93c5fd", "#e5e7eb"];

/**
 * Build a 12×31 category matrix for the presence heatmap.
 *
 * Each row is a month; the first `min` columns are "certain home" (0), the
 * columns up to `max` are "possible visit" (1), the rest up to the month
 * length are "away" (2), and days beyond the month length are empty (-1).
 * It is a schematic of the *band* (counts), not specific calendar dates —
 * the simulator scatters the home days at random within each month.
 */
export function presenceHeatmapMatrix(cal) {
    const { min, max } = calendarMinMax(cal);
    const matrix = [];
    for (let m = 0; m < 12; m++) {
        const L = MONTH_LENGTHS[m];
        const row = [];
        for (let day = 0; day < 31; day++) {
            if (day >= L) row.push(-1);
            else if (day < min[m]) row.push(0);
            else if (day < max[m]) row.push(1);
            else row.push(2);
        }
        matrix.push(row);
    }
    return matrix;
}

/**
 * Client mirror of fit_bolletta_profile: the ARERA home-scale factor that
 * makes the expected annual energy match `targetAnnualKwh`, given occupancy.
 * Returns the same keys as the backend fit so the KPI can show the split
 * before the (server-side) preview returns.
 */
export function fitBollettaClient(targetAnnualKwh, cal) {
    const base = ARERA_BASELINE_ANNUAL_KWH;
    const f = annualPresenceFraction(cal);
    const awayKwh = base * (1 - f);
    if (f <= 1e-9) {
        return {
            home_scale_factor: 1.0,
            estimated_home_kwh: 0.0,
            estimated_away_kwh: base,
            annual_presence_fraction: f,
        };
    }
    let homeScale = (targetAnnualKwh - awayKwh) / (base * f);
    homeScale = Math.max(homeScale, MIN_HOME_SCALE_FACTOR);
    return {
        home_scale_factor: homeScale,
        estimated_home_kwh: f * homeScale * base,
        estimated_away_kwh: awayKwh,
        annual_presence_fraction: f,
    };
}

/** Assemble the savable `data` block for a bill-fitted profile. */
export function buildBollettaData(annualKwh, houseType, cal, fit) {
    return {
        kind: "bolletta",
        input_level: "bolletta",
        presence_calendar: normalizeCalendar(cal),
        bolletta: { annual_kwh: Number(annualKwh), house_type: houseType || null },
        _derived: {
            home_scale_factor: fit.home_scale_factor,
            estimated_home_kwh: fit.estimated_home_kwh,
            estimated_away_kwh: fit.estimated_away_kwh,
            annual_presence_fraction: fit.annual_presence_fraction,
        },
    };
}

// ── Named starting points (mirror PRESENCE_CALENDAR_PRESETS on the backend) ──

function uniform(pattern) {
    return { months: Array.from({ length: 12 }, () => ({ ...pattern })) };
}

export const PRESENCE_PRESETS = {
    primary_residence: {
        label: "Residenza principale",
        calendar: uniform({ weekends: true, full_weeks: 3, extra_weekdays: 2, visit_probability: 0.0 }),
    },
    summer_vacation: {
        label: "Seconda casa estiva",
        calendar: {
            months: Array.from({ length: 12 }, (_, i) =>
                i >= 5 && i <= 7
                    ? { weekends: true, full_weeks: 4, extra_weekdays: 0, visit_probability: 0.0 }
                    : { weekends: true, full_weeks: 0, extra_weekdays: 0, visit_probability: 0.1 },
            ),
        },
    },
    weekend_house: {
        label: "Casa weekend",
        calendar: uniform({ weekends: true, full_weeks: 0, extra_weekdays: 0, visit_probability: 0.0 }),
    },
};

/** Default presence calendar (~23 occupied days/month primary residence). */
export function defaultPresenceCalendar() {
    return uniform({ weekends: false, full_weeks: 3, extra_weekdays: 2, visit_probability: 0.0 });
}
