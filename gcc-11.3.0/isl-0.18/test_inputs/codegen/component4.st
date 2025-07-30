domain: "{ A[i] : 0 <= i < 10; B[i,j] : 0 <= i,j < 10 }"
child:
  schedule: "[{ A[i] -> [0]; B[i,j] -> [i] }]"
  child:
    sequence:
    - filter: "{ A[i] }"
    - filter: "{ B[i,j] }"
