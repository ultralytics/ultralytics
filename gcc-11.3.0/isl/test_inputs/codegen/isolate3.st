# Check use of options specific to isolated part
domain: "{ A[i] : 0 <= i < 100 }"
child:
  schedule: "[{ A[i] -> [i] }]"
  options: "{ isolate[[] -> [x]] : 10 <= x <= 20; [isolate[] -> unroll[x]] }"
