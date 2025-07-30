domain: "{ a[i] : 0 <= i < 10; b[i] : 0 <= i < 10 }"
child:
  schedule: "[{ a[i] -> [i]; b[i] -> [i+1] }]"
  options: "{ atomic[x] }"
