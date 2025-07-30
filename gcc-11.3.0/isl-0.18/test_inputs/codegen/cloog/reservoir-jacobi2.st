domain: "[M] -> { S1[i0, i1] : i0 >= 0 and i0 <= -1 + M and i1 >= 0 and i1 <= -1 + M }"
child:
  context: "[M] -> { [] : M >= 1 }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      schedule: "[M] -> [{ S1[i0, i1] -> [(i1)] }]"
      options: "[M] -> { separate[i0] }"
