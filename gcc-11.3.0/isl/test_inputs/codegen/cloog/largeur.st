domain: "[M] -> { S1[i0, i1] : i0 >= 1 and i0 <= M and i1 >= i0 and i1 <= M }"
child:
  context: "[M] -> { [] : M >= 0 }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i1)] }, { S1[i0, i1] -> [(i0)] }]"
    options: "[M] -> { separate[i0] }"
