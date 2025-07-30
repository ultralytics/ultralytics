domain: "[M] -> { S1[i0, i1] : 7i1 = -2 + i0 and i0 >= 2 and i0 <= M }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)] }]"
    options: "[M] -> { separate[i0] }"
