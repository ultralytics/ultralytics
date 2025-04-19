// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

!function(){function r(t,e){if(!(this instanceof r))return new r(t,e);if(!t||"TABLE"!==t.tagName)throw new Error("Element must be a table");this.init(t,e||{})}function v(t){var e;return window.CustomEvent&&"function"==typeof window.CustomEvent?e=new CustomEvent(t):(e=document.createEvent("CustomEvent")).initCustomEvent(t,!1,!1,void 0),e}function p(t,e){return t.getAttribute(e.sortAttribute||"data-sort")||t.textContent||t.innerText||""}function A(t,e){return(t=t.trim().toLowerCase())===(e=e.trim().toLowerCase())?0:t<e?1:-1}function E(t,e){return[].slice.call(t).find(function(t){return t.getAttribute("data-sort-column-key")===e})}function x(n,o){return function(t,e){var r=n(t.td,e.td);return 0===r?o?e.index-t.index:t.index-e.index:r}}var B=[];r.extend=function(t,e,r){if("function"!=typeof e||"function"!=typeof r)throw new Error("Pattern and sort must be a function");B.push({name:t,pattern:e,sort:r})},r.prototype={init:function(t,e){var r,n,o,i=this;if(i.table=t,i.thead=!1,i.options=e,t.rows&&0<t.rows.length)if(t.tHead&&0<t.tHead.rows.length){for(a=0;a<t.tHead.rows.length;a++)if("thead"===t.tHead.rows[a].getAttribute("data-sort-method")){r=t.tHead.rows[a];break}r=r||t.tHead.rows[t.tHead.rows.length-1],i.thead=!0}else r=t.rows[0];if(r){function s(){i.current&&i.current!==this&&i.current.removeAttribute("aria-sort"),i.current=this,i.sortTable(this)}for(var a=0;a<r.cells.length;a++)(o=r.cells[a]).setAttribute("role","columnheader"),"none"!==o.getAttribute("data-sort-method")&&(o.tabindex=0,o.addEventListener("click",s,!1),null!==o.getAttribute("data-sort-default")&&(n=o));n&&(i.current=n,i.sortTable(n))}},sortTable:function(t,e){var r=this,n=t.getAttribute("data-sort-column-key"),o=t.cellIndex,i=A,s="",a=[],d=r.thead?0:1,u=t.getAttribute("data-sort-method"),l=t.hasAttribute("data-sort-reverse"),c=t.getAttribute("aria-sort");if(r.table.dispatchEvent(v("beforeSort")),e||(c="ascending"===c||"descending"!==c&&r.options.descending?"descending":"ascending",t.setAttribute("aria-sort",c)),!(r.table.rows.length<2)){if(!u){for(;a.length<3&&d<r.table.tBodies[0].rows.length;)0<(s=(s=(h=n?E(r.table.tBodies[0].rows[d].cells,n):r.table.tBodies[0].rows[d].cells[o])?p(h,r.options):"").trim()).length&&a.push(s),d++;if(!a)return}for(d=0;d<B.length;d++)if(s=B[d],u){if(s.name===u){i=s.sort;break}}else if(a.every(s.pattern)){i=s.sort;break}for(r.col=o,d=0;d<r.table.tBodies.length;d++){var f,h,b=[],g={},w=0,m=0;if(!(r.table.tBodies[d].rows.length<2)){for(f=0;f<r.table.tBodies[d].rows.length;f++)"none"===(s=r.table.tBodies[d].rows[f]).getAttribute("data-sort-method")?g[w]=s:(h=n?E(s.cells,n):s.cells[r.col],b.push({tr:s,td:h?p(h,r.options):"",index:w})),w++;for("descending"===c&&!l||"ascending"===c&&l?b.sort(x(i,!0)):(b.sort(x(i,!1)),b.reverse()),f=0;f<w;f++)g[f]?(s=g[f],m++):s=b[f-m].tr,r.table.tBodies[d].appendChild(s)}}r.table.dispatchEvent(v("afterSort"))}},refresh:function(){void 0!==this.current&&this.sortTable(this.current,!0)}},"undefined"!=typeof module&&module.exports?module.exports=r:window.Tablesort=r}();

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
