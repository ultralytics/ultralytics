domain: "[M, N] -> { S1[i0, i1, i2] : i0 >= 1 and i1 >= 0 and i1 <= M and i1 <= i0 and i1 <= N - i0 and i2 >= 0 and i2 <= M and i2 <= i0 and i2 <= N - i0 }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1, i2] -> [(i0)] }, { S1[i0, i1, i2] -> [(i1)] }, { S1[i0, i1, i2] -> [(i2)] }]"
    options: "[M, N] -> { separate[i0] }"
