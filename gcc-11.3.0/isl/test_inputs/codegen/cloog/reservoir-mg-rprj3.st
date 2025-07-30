domain: "[M, N, O, P, Q, R] -> { S2[i0, i1, i2] : i0 >= 2 and i0 <= -1 + O and i1 >= 2 and i1 <= -1 + N and i2 >= 2 and i2 <= M; S4[i0, i1, i2] : i0 >= 2 and i0 <= -1 + O and i1 >= 2 and i1 <= -1 + N and i2 >= 2 and i2 <= -1 + M; S1[i0, i1, i2] : i0 >= 2 and i0 <= -1 + O and i1 >= 2 and i1 <= -1 + N and i2 >= 2 and i2 <= M; S5[i0, i1, i2] : i0 >= 2 and i0 <= -1 + O and i1 >= 2 and i1 <= -1 + N and i2 >= 2 and i2 <= -1 + M; S3[i0, i1, i2] : i0 >= 2 and i0 <= -1 + O and i1 >= 2 and i1 <= -1 + N and i2 >= 2 and i2 <= -1 + M }"
child:
  context: "[M, N, O, P, Q, R] -> { [] }"
  child:
    schedule: "[M, N, O, P, Q, R] -> [{ S5[i0, i1, i2] -> [(i0)]; S3[i0, i1, i2] -> [(i0)]; S4[i0, i1, i2] -> [(i0)]; S1[i0, i1, i2] -> [(i0)]; S2[i0, i1, i2] -> [(i0)] }]"
    options: "[M, N, O, P, Q, R] -> { separate[i0] }"
    child:
      schedule: "[M, N, O, P, Q, R] -> [{ S5[i0, i1, i2] -> [(1 + i1)]; S3[i0, i1, i2] -> [(1 + i1)]; S4[i0, i1, i2] -> [(1 + i1)]; S1[i0, i1, i2] -> [(i1)]; S2[i0, i1, i2] -> [(1 + i1)] }]"
      options: "[M, N, O, P, Q, R] -> { separate[i0] }"
      child:
        sequence:
        - filter: "[M, N, O, P, Q, R] -> { S2[i0, i1, i2] }"
          child:
            schedule: "[M, N, O, P, Q, R] -> [{ S2[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O, P, Q, R] -> { separate[i0] }"
        - filter: "[M, N, O, P, Q, R] -> { S4[i0, i1, i2]; S5[i0, i1, i2]; S3[i0, i1, i2] }"
          child:
            schedule: "[M, N, O, P, Q, R] -> [{ S5[i0, i1, i2] -> [(i2)]; S3[i0, i1, i2] -> [(i2)]; S4[i0, i1, i2] -> [(-1 + i2)] }]"
            options: "[M, N, O, P, Q, R] -> { separate[i0] }"
            child:
              sequence:
              - filter: "[M, N, O, P, Q, R] -> { S3[i0, i1, i2] }"
              - filter: "[M, N, O, P, Q, R] -> { S5[i0, i1, i2] }"
              - filter: "[M, N, O, P, Q, R] -> { S4[i0, i1, i2] }"
        - filter: "[M, N, O, P, Q, R] -> { S1[i0, i1, i2] }"
          child:
            schedule: "[M, N, O, P, Q, R] -> [{ S1[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O, P, Q, R] -> { separate[i0] }"
