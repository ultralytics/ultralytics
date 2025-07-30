domain: "[M] -> { S1[1, i1] : 2i1 >= M and 2i1 <= 1 + M }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)] }]"
    options: "[M] -> { separate[i0] }"
