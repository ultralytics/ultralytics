domain: "[M, N, O] -> { S1[i0] : i0 >= -M and i0 >= -N and i0 <= N and i0 <= O }"
child:
  context: "[M, N, O] -> { [] }"
  child:
    schedule: "[M, N, O] -> [{ S1[i0] -> [(i0)] }]"
    options: "[M, N, O] -> { separate[i0] }"
