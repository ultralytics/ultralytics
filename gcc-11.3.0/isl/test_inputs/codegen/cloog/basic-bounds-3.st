domain: "[M] -> { S1[i0] : i0 >= 0 and i0 <= M }"
child:
  context: "[M] -> { [] : M >= 0 }"
  child:
    schedule: "[M] -> [{ S1[i0] -> [(i0)] }]"
    options: "[M] -> { separate[i0] }"
