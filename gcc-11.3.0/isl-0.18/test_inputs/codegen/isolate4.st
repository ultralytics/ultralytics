# Check that generic options are not applied to isolated part
domain: "{ A[i] : 0 <= i < 20 }"
child:
  schedule: "[{ A[i] -> [i] }]"
  options: "{ isolate[[] -> [x]] : 5 <= x <= 15; unroll[x] }"
