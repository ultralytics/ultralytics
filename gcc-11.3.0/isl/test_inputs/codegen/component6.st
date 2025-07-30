# Check that components are still detected in presence of nested context node
domain: "{ A[]; B[i] : 0 <= i < 10 }"
child:
  schedule: "[{ A[] -> [0]; B[i] -> [i] }]"
  child:
    context: "[n] -> { [i] : 0 <= n <= i }"
    child:
      sequence:
      - filter: "{ A[] }"
      - filter: "{ B[i] }"
