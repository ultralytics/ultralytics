# Check that options are interpreted locally
domain: "{ A[i,j,k] : 0 <= i,k < 100 and 0 <= j < 2; B[i,j,k] : 0 <= i,k < 100 and 0 <= j < 2 }"
child:
  schedule: "[{ A[i,j,k] -> [k]; B[i,j,k] -> [k] }]"
  child:
    schedule: "[{ A[i,j,k] -> [2i]; B[i,j,k] -> [2i+1] }, { A[i,j,k] -> [j]; B[i,j,k] -> [j]}]"
    options: "{ unroll[1] }"
