domain: "[n, m] -> { S1[i0] : i0 >= 1 and i0 <= m; S2[i0, i1] : i0 >= 1 and i0 <= m and i1 >= 1 and i1 <= n }"
child:
  context: "[n, m] -> { [] }"
  child:
    schedule: "[n, m] -> [{ S2[i0, i1] -> [(i0)]; S1[i0] -> [(i0)] }, { S2[i0, i1] -> [(i1)]; S1[i0] -> [(0)] }]"
    options: "[n, m] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[n, m] -> { S1[i0] }"
      - filter: "[n, m] -> { S2[i0, i1] }"
