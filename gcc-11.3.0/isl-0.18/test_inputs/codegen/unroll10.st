# Check that all information is taken into account while trying to unroll
domain: "[m,n] -> { A[i] : 0 <= i < n,m }"
child:
  context: "[m,n] -> { [] : m <= 10 or n <= 10 }"
  child:
    schedule: "[{ A[i] -> [i] }]"
    options: "{ unroll[x] }"
