domain: "[n] -> { S1[i0, i1] : i0 >= 0 and i0 <= n and i1 >= 0 and i1 <= -15 + n; S2[i0, i1] : i0 >= 15 and i0 <= n and i1 >= 10 and i1 <= n }"
child:
  context: "[n] -> { [] : n >= 25 }"
  child:
    schedule: "[n] -> [{ S2[i0, i1] -> [(i0)]; S1[i0, i1] -> [(i0)] }, { S2[i0, i1] -> [(i1)]; S1[i0, i1] -> [(i1)] }]"
    options: "[n] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[n] -> { S1[i0, i1] }"
      - filter: "[n] -> { S2[i0, i1] }"
