domain: "[M] -> { S1[i0] : i0 >= 0 and i0 <= 100 }"
child:
  context: "[M] -> { [] : M >= 1 or M <= -1 }"
  child:
    schedule: "[M] -> [{ S1[i0] -> [(i0)] : M <= 10; S1[i0] -> [(-i0)] : M >= 11 }]"
    options: "[M] -> { separate[i0] }"
