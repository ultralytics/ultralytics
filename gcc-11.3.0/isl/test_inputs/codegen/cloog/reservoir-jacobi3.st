domain: "[M, N] -> { S2[i0, i1, i2] : i0 >= 1 and i0 <= M and i1 >= 2 and i1 <= -1 + N and i2 >= 2 and i2 <= -1 + N; S1[i0, i1, i2] : i0 >= 1 and i0 <= M and i1 >= 2 and i1 <= -1 + N and i2 >= 2 and i2 <= -1 + N }"
child:
  context: "[M, N] -> { [] }"
  child:
    schedule: "[M, N] -> [{ S1[i0, i1, i2] -> [(2i0)]; S2[i0, i1, i2] -> [(1 + 2i0)] }]"
    options: "[M, N] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S1[i0, i1, i2] }"
        child:
          schedule: "[M, N] -> [{ S1[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
          child:
            schedule: "[M, N] -> [{ S1[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N] -> { separate[i0] }"
      - filter: "[M, N] -> { S2[i0, i1, i2] }"
        child:
          schedule: "[M, N] -> [{ S2[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N] -> { separate[i0] }"
          child:
            schedule: "[M, N] -> [{ S2[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N] -> { separate[i0] }"
