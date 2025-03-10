// tablesort.filesize.min.js
!(function () {
  function r(t) {
    return (
      (t = t.match(/^(\d+(\.\d+)?) ?((K|M|G|T|P|E|Z|Y|B$)i?B?)$/i)),
      parseFloat(t[1].replace(/[^\-?0-9.]/g, "")) *
        (function (t) {
          var e = "i" === (t = t.toLowerCase())[1] ? 1024 : 1e3;
          switch (t[0]) {
            case "k":
              return Math.pow(e, 2);
            case "m":
              return Math.pow(e, 3);
            case "g":
              return Math.pow(e, 4);
            case "t":
              return Math.pow(e, 5);
            case "p":
              return Math.pow(e, 6);
            case "e":
              return Math.pow(e, 7);
            case "z":
              return Math.pow(e, 8);
            case "y":
              return Math.pow(e, 9);
            default:
              return e;
          }
        })(t[3])
    );
  }
  Tablesort.extend(
    "filesize",
    function (t) {
      return /^\d+(\.\d+)? ?(K|M|G|T|P|E|Z|Y|B$)i?B?$/i.test(t);
    },
    function (t, e) {
      return (
        (t = r(t)),
        (e = r(e)),
        (e = e),
        (t = t),
        (e = parseFloat(e)),
        (t = parseFloat(t)),
        (e = isNaN(e) ? 0 : e) - (t = isNaN(t) ? 0 : t)
      );
    },
  );
})();

// tablesort.dotsep.min.js
Tablesort.extend(
  "dotsep",
  function (t) {
    return /^(\d+\.)+\d+$/.test(t);
  },
  function (t, r) {
    (t = t.split(".")), (r = r.split("."));
    for (var e, n, i = 0, s = t.length; i < s; i++)
      if ((e = parseInt(t[i], 10)) !== (n = parseInt(r[i], 10))) {
        if (n < e) return -1;
        if (e < n) return 1;
      }
    return 0;
  },
);

// tablesort.number.min.js
(function () {
  var cleanNumber = function (i) {
      // Remove everything after ± symbol if present
      i = i.split("±")[0].trim();
      return i.replace(/[^\-?0-9.]/g, "");
    },
    compareNumber = function (a, b) {
      a = parseFloat(a);
      b = parseFloat(b);

      a = isNaN(a) ? 0 : a;
      b = isNaN(b) ? 0 : b;

      return a - b;
    };

  Tablesort.extend(
    "number",
    function (item) {
      return (
        item.match(/^[-+]?[£\x24Û¢´€]?\d+\s*([,\.]\d{0,2})/) || // Prefixed currency
        item.match(/^[-+]?\d+\s*([,\.]\d{0,2})?[£\x24Û¢´€]/) || // Suffixed currency
        item.match(/^[-+]?(\d)*-?([,\.]){0,1}-?(\d)+([E,e][\-+][\d]+)?%?$/)
      ); // Number
    },
    function (a, b) {
      a = cleanNumber(a);
      b = cleanNumber(b);

      return compareNumber(b, a);
    },
  );
})();

// subscribe
document$.subscribe(function () {
  var tables = document.querySelectorAll("article table:not([class])");
  tables.forEach(function (table) {
    new Tablesort(table);
  });
});
