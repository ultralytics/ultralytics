domain: "[M] -> { S1[i0, i1] : i0 >= 0 and i0 <= M and i0 <= 10 and i1 >= 0 and i1 <= M and i1 <= 10 }"
child:
  context: "[M] -> { [] : M >= 0 }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)] }]"
    options: "[M] -> { separate[i0] }"
