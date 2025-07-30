# Check that the isolate option is adjusted by schedule space scaling
domain: "{ A[i] : 0 <= i < 100 }"
child:
  schedule: "[{ A[i] -> [3i] }]"
  options: "{ isolate[[] -> [x]] : 10 <= x <= 20 }"
