domain: "[M, N, O, P, Q, R, S, T, U] -> { S1[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= Q and i1 <= -1 + N and i2 >= P and i2 <= -1 + M; S3[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= P and i2 <= -1 + M; S4[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S2[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= Q and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M }"
child:
  context: "[M, N, O, P, Q, R, S, T, U] -> { [] : M >= 10 and N >= 10 and O >= 10 and P >= 1 and P <= 2 and Q >= 1 and Q <= 2 and R >= 1 and R <= 2 and S >= 0 and S <= 1 and T >= 0 and T <= 1 and U >= 0 and U <= 1 }"
  child:
    schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S4[i0, i1, i2] -> [(i0)]; S2[i0, i1, i2] -> [(i0)]; S1[i0, i1, i2] -> [(i0)]; S3[i0, i1, i2] -> [(i0)] }]"
    options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M, N, O, P, Q, R, S, T, U] -> { S1[i0, i1, i2]; S2[i0, i1, i2] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S2[i0, i1, i2] -> [(i1)]; S1[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            sequence:
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S1[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S1[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S2[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S2[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
      - filter: "[M, N, O, P, Q, R, S, T, U] -> { S3[i0, i1, i2]; S4[i0, i1, i2] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S4[i0, i1, i2] -> [(i1)]; S3[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            sequence:
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S3[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S3[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S4[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S4[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
