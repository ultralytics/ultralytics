domain: "{ A[]; B[i] : 0 <= i < 10 }"
child:
  schedule: "[{ A[] -> [0]; B[i] -> [i] }]"
  child:
    sequence:
    - filter: "{ A[] }"
    - filter: "{ B[i] }"
