domain: "[m, n, p, q] -> { S1[i0] : i0 >= m and i0 <= n; S2[i0] : i0 >= p and i0 <= q }"
child:
  context: "[m, n, p, q] -> { [] }"
  child:
    schedule: "[m, n, p, q] -> [{ S2[i0] -> [(i0)]; S1[i0] -> [(i0)] }]"
    options: "[m, n, p, q] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[m, n, p, q] -> { S1[i0] }"
      - filter: "[m, n, p, q] -> { S2[i0] }"
