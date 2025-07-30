domain: "[m, n] -> { S1[i0, i1] : i0 >= 1 and i1 <= m and i1 >= -1 + i0; S2[i0, i1] : i0 >= 2 and i0 <= n and i1 >= 0 and i1 <= n }"
child:
  context: "[m, n] -> { [] : n <= m and m >= 0 and n >= 2 }"
  child:
    schedule: "[m, n] -> [{ S2[i0, i1] -> [(i0)]; S1[i0, i1] -> [(i0)] }, { S2[i0, i1] -> [(i1)]; S1[i0, i1] -> [(i1)] }]"
    options: "[m, n] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[m, n] -> { S1[i0, i1] }"
      - filter: "[m, n] -> { S2[i0, i1] }"
