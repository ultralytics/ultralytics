# Check that options are adjusted by shifted stride detection
domain: "{ A[i,j] : 0 <= i < 100 and 0 <= j < 2; B[i,j] : 0 <= i < 100 and 0 <= j < 2 }"
child:
  schedule: "[{ A[i,j] -> [2i]; B[i,j] -> [2i+1] }, { A[i,j] -> [j]; B[i,j] -> [j]}]"
  options: "{ unroll[1] }"
