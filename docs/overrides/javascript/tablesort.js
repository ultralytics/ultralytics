document$.subscribe(function () {
  var tables = document.querySelectorAll("article div.tip table");
  tables.forEach(function (table) {
    new Tablesort(table);
  });
});
