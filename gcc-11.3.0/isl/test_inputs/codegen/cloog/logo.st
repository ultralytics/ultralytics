domain: "[M] -> { S1[i0, i1] : i0 >= 1 and i1 <= 7 and i1 >= -1 + i0; S2[i0, i1] : i0 >= 2 and i0 <= 6 and i1 >= 0 and i1 <= 4 }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S1[i0, i1] }"
      - filter: "[M] -> { S2[i0, i1] }"
