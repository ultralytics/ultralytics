// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// tablesort.filesize.min.js
!(function () {
  const filesizeRegex = /^(\d+(\.\d+)?) ?((K|M|G|T|P|E|Z|Y|B$)i?B?)$/i;

  function r(t) {
    t = t.match(filesizeRegex);
    if (!t) {
      return 0;
    }

    const value = parseFloat(t[1].replace(/[^\-?0-9.]/g, ""));
    const unit = t[3].toLowerCase();
    const base = unit[1] === "i" ? 1024 : 1e3;
    const powers = { k: 2, m: 3, g: 4, t: 5, p: 6, e: 7, z: 8, y: 9 };

    return value * (powers[unit[0]] ? Math.pow(base, powers[unit[0]]) : base);
  }

  Tablesort.extend(
    "filesize",
    (t) => filesizeRegex.test(t),
    (t, e) => {
      t = r(t);
      e = r(e);
      return (isNaN(e) ? 0 : e) - (isNaN(t) ? 0 : t);
    },
  );
})();

// tablesort.dotsep.min.js
Tablesort.extend(
  "dotsep",
  (t) => /^(\d+\.)+\d+$/.test(t),
  (t, r) => {
    t = t.split(".");
    r = r.split(".");

    for (let i = 0, s = t.length; i < s; i++) {
      const e = parseInt(t[i], 10);
      const n = parseInt(r[i], 10);

      if (e !== n) {
        return n < e ? -1 : 1;
      }
    }
    return 0;
  },
);

// tablesort.number.min.js
(function () {
  const cleanNumber = (i) =>
    i
      .split("±")[0]
      .trim()
      .replace(/[^\-?0-9.]/g, "");
  const compareNumber = (a, b) => (parseFloat(a) || 0) - (parseFloat(b) || 0);

  Tablesort.extend(
    "number",
    (item) =>
      item.match(/^[-+]?[£\x24Û¢´€]?\d+\s*([,\.]\d{0,2})/) || // Prefixed currency
      item.match(/^[-+]?\d+\s*([,\.]\d{0,2})?[£\x24Û¢´€]/) || // Suffixed currency
      item.match(/^[-+]?(\d)*-?([,\.]){0,1}-?(\d)+([E,e][\-+][\d]+)?%?$/), // Number
    (a, b) => compareNumber(cleanNumber(b), cleanNumber(a)),
  );
})();

// subscribe
document$.subscribe(() => {
  document.querySelectorAll("article table:not([class])").forEach((table) => {
    new Tablesort(table);
  });
});
