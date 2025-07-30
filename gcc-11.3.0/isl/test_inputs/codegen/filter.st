# Check proper handling of filters that turn out to be empty on their paths
domain: "[n,m] -> { A[i,j] : 0 <= i,j < n }"
child:
  set:
  - filter: "[n,m] -> { A[i,j] : m < n }"
    child:
      schedule: "[{ A[i,j] -> [i] }]"
      child:
        set:
        - filter: "[n,m] -> { A[i,j] : m < n }"
        - filter: "[n,m] -> { A[i,j] : m >= n }"
  - filter: "[n,m] -> { A[i,j] : m >= n }"
    child:
      schedule: "[{ A[i,j] -> [i] }]"
      child:
        set:
        - filter: "[n,m] -> { A[i,j] : m < n }"
        - filter: "[n,m] -> { A[i,j] : m >= n }"
