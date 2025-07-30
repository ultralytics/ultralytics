domain: "[M, N] -> { S2[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= N; S1[i0, i1, i2] : i0 >= 1 and i0 <= M and i1 <= N and i2 >= 1 and i2 <= -1 + i1 }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S2[i0, i1] -> [(i0)]; S1[i0, i1, i2] -> [(i0)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      schedule: "[M, N] -> [{ S2[i0, i1] -> [(-1 + 2i1)]; S1[i0, i1, i2] -> [(i1 + i2)] }]"
      options: "[M, N] -> { separate[i0] }"
      child:
        sequence:
        - filter: "[M, N] -> { S1[i0, i1, i2] }"
          child:
            schedule: "[M, N] -> [{ S1[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N] -> { separate[i0] }"
        - filter: "[M, N] -> { S2[i0, i1] }"
