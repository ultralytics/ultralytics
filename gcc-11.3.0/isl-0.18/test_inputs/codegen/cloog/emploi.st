domain: "[m, n] -> { S1[i0] : (i0 >= 1 and i0 <= n and i0 <= 2m) or (i0 >= m and i0 >= 1 and i0 <= n); S2[i0, i1] : i0 >= 1 and i0 <= n and i1 >= 1 and i1 <= m }"
child:
  context: "[m, n] -> { [] }"
  child:
    schedule: "[m, n] -> [{ S1[i0] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S1[i0] -> [(0)]; S2[i0, i1] -> [(i1)] }]"
    options: "[m, n] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[m, n] -> { S1[i0] }"
      - filter: "[m, n] -> { S2[i0, i1] }"
