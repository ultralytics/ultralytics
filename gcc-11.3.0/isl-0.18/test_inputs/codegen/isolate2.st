# Check that the isolate option is adjusted by schedule space scaling
domain: "{ A[i,j] : 0 <= i,j < 100 }"
child:
  schedule: "[{ A[i,j] -> [3i] }]"
  child:
    schedule: "[{ A[i,j] -> [3j] }]"
    options: "{ isolate[[x] -> [y]] : 10 <= x <= 20 }"
