domain: "[M, N, O] -> { S3[i0, i1, i2] : i0 >= 2 and i0 <= -1 + O and i1 >= 2 and i1 <= -1 + N and i2 >= 2 and i2 <= -1 + M; S2[i0, i1, i2] : i0 >= 2 and i0 <= -1 + O and i1 >= 2 and i1 <= -1 + N and i2 >= 1 and i2 <= M; S1[i0, i1, i2] : i0 >= 2 and i0 <= -1 + O and i1 >= 2 and i1 <= -1 + N and i2 >= 1 and i2 <= M }"
child:
  context: "[M, N, O] -> { [] }"
  child:
    schedule: "[M, N, O] -> [{ S3[i0, i1, i2] -> [(i0)]; S2[i0, i1, i2] -> [(i0)]; S1[i0, i1, i2] -> [(i0)] }]"
    options: "[M, N, O] -> { separate[i0] }"
    child:
      schedule: "[M, N, O] -> [{ S3[i0, i1, i2] -> [(2i1)]; S2[i0, i1, i2] -> [(-1 + 2i1)]; S1[i0, i1, i2] -> [(-1 + 2i1)] }]"
      options: "[M, N, O] -> { separate[i0] }"
      child:
        sequence:
        - filter: "[M, N, O] -> { S2[i0, i1, i2]; S1[i0, i1, i2] }"
          child:
            schedule: "[M, N, O] -> [{ S2[i0, i1, i2] -> [(i2)]; S1[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O] -> { separate[i0] }"
            child:
              sequence:
              - filter: "[M, N, O] -> { S1[i0, i1, i2] }"
              - filter: "[M, N, O] -> { S2[i0, i1, i2] }"
        - filter: "[M, N, O] -> { S3[i0, i1, i2] }"
          child:
            schedule: "[M, N, O] -> [{ S3[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O] -> { separate[i0] }"
