domain: "[n] -> { S1[i0, i1] : 2i1 = i0 and i0 >= 1 and i0 <= n; S2[i0, i1] : 4i1 = i0 and i0 >= 1 and i0 <= n }"
child:
  context: "[n] -> { [] : n >= 2 }"
  child:
    schedule: "[n] -> [{ S2[i0, i1] -> [(i0)]; S1[i0, i1] -> [(i0)] }, { S2[i0, i1] -> [(i1)]; S1[i0, i1] -> [(i1)] }]"
    options: "[n] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[n] -> { S1[i0, i1] }"
      - filter: "[n] -> { S2[i0, i1] }"
