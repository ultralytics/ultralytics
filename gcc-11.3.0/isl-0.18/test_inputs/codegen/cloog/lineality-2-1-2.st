domain: "[M] -> { S1[i0, i1] : i0 >= 1 and i1 >= 1 and i0 <= M and i1 <= M; S2[i0, 2 + i0] : i0 >= 1 and i0 <= M }"
child:
  context: "[M] -> { [] : M >= 2 }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S1[i0, i1] }"
      - filter: "[M] -> { S2[i0, i1] }"
