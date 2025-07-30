domain: "[M] -> { S1[i0, i1] : i0 >= 1 and i0 <= 10 and i1 >= 1 and i1 <= i0; S2[i0, i1] : i0 >= 1 and i0 <= 10 and i1 >= 11 and i1 <= M }"
child:
  context: "[M] -> { [] : M >= 20 }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      schedule: "[M] -> [{ S1[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
      options: "[M] -> { separate[i0] }"
