domain: "[m, n, p] -> { S1[i, k, j] : 2k = -1 + i and i >= 1 and i <= m and j >= 1 and j <= p; S2[i, k, j] : 2k = -1 + i and i >= 1 and i <= n and j >= 1 and j <= i }"
child:
  context: "[m, n, p] -> { [] : n = 6 and m >= 7 and p >= 7 }"
  child:
    schedule: "[m, n, p] -> [{ S1[i, k, j] -> [(i)]; S2[i, k, j] -> [(i)] }, { S1[i, k, j] -> [(k)]; S2[i, k, j] -> [(k)] }, { S1[i, k, j] -> [(j)]; S2[i, k, j] -> [(j)] }]"
    options: "[m, n, p] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[m, n, p] -> { S1[i, k, j] }"
      - filter: "[m, n, p] -> { S2[i, k, j] }"
