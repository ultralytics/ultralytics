domain: "[n, m] -> { S1[i0, i0] : i0 >= 1 and i0 <= n; S3[i0, n] : i0 >= 1 and i0 <= m; S2[i0, i1] : i0 >= 1 and i0 <= n and i1 >= i0 and i1 <= n }"
child:
  context: "[n, m] -> { [] : n >= 2 and m >= n }"
  child:
    schedule: "[n, m] -> [{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)]; S3[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)]; S3[i0, i1] -> [(i1)] }]"
    options: "[n, m] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[n, m] -> { S1[i0, i1] }"
      - filter: "[n, m] -> { S2[i0, i1] }"
      - filter: "[n, m] -> { S3[i0, i1] }"
