domain: "[M, N, O, P, Q, R, S, T, U] -> { S8[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S12[i0, i1, i2] : i0 >= R and i0 <= -1 + O and i1 >= Q and i1 <= -1 + N and i2 >= P and i2 <= -1 + M; S5[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S10[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S6[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S1[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= M; S3[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= M; S4[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S15[i0, i1, i2] : i0 >= R and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S11[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S2[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= M; S7[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S9[i0, i1, i2] : i0 >= 1 and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M; S14[i0, i1, i2] : i0 >= R and i0 <= -1 + O and i1 >= 1 and i1 <= -1 + N and i2 >= P and i2 <= -1 + M; S13[i0, i1, i2] : i0 >= R and i0 <= -1 + O and i1 >= Q and i1 <= -1 + N and i2 >= 1 and i2 <= -1 + M }"
child:
  context: "[M, N, O, P, Q, R, S, T, U] -> { [] }"
  child:
    sequence:
    - filter: "[M, N, O, P, Q, R, S, T, U] -> { S10[i0, i1, i2]; S6[i0, i1, i2]; S3[i0, i1, i2]; S1[i0, i1, i2]; S11[i0, i1, i2]; S7[i0, i1, i2] }"
      child:
        schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S11[i0, i1, i2] -> [(i0)]; S1[i0, i1, i2] -> [(i0)]; S6[i0, i1, i2] -> [(i0)]; S10[i0, i1, i2] -> [(i0)]; S3[i0, i1, i2] -> [(i0)]; S7[i0, i1, i2] -> [(i0)] }]"
        options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S11[i0, i1, i2] -> [(2i1)]; S1[i0, i1, i2] -> [(-3 + 2i1)]; S6[i0, i1, i2] -> [(-2 + 2i1)]; S10[i0, i1, i2] -> [(1 + 2i1)]; S3[i0, i1, i2] -> [(-1 + 2i1)]; S7[i0, i1, i2] -> [(-2 + 2i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            sequence:
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S10[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S10[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S3[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S3[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S6[i0, i1, i2]; S1[i0, i1, i2]; S7[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S1[i0, i1, i2] -> [(i2)]; S6[i0, i1, i2] -> [(i2)]; S7[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
                child:
                  sequence:
                  - filter: "[M, N, O, P, Q, R, S, T, U] -> { S6[i0, i1, i2]; S1[i0, i1, i2] }"
                  - filter: "[M, N, O, P, Q, R, S, T, U] -> { S7[i0, i1, i2] }"
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S11[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S11[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
    - filter: "[M, N, O, P, Q, R, S, T, U] -> { S2[i0, i1, i2]; S9[i0, i1, i2]; S8[i0, i1, i2] }"
      child:
        schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S2[i0, i1, i2] -> [(i0)]; S8[i0, i1, i2] -> [(i0)]; S9[i0, i1, i2] -> [(i0)] }]"
        options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S2[i0, i1, i2] -> [(2i1)]; S8[i0, i1, i2] -> [(2i1)]; S9[i0, i1, i2] -> [(1 + 2i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            sequence:
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S2[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S2[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S8[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S8[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
            - filter: "[M, N, O, P, Q, R, S, T, U] -> { S9[i0, i1, i2] }"
              child:
                schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S9[i0, i1, i2] -> [(i2)] }]"
                options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
    - filter: "[M, N, O, P, Q, R, S, T, U] -> { S4[i0, i1, i2] }"
      child:
        schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S4[i0, i1, i2] -> [(i0)] }]"
        options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S4[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S4[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
    - filter: "[M, N, O, P, Q, R, S, T, U] -> { S5[i0, i1, i2] }"
      child:
        schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S5[i0, i1, i2] -> [(i0)] }]"
        options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S5[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S5[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
    - filter: "[M, N, O, P, Q, R, S, T, U] -> { S12[i0, i1, i2] }"
      child:
        schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S12[i0, i1, i2] -> [(i0)] }]"
        options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S12[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S12[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
    - filter: "[M, N, O, P, Q, R, S, T, U] -> { S13[i0, i1, i2] }"
      child:
        schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S13[i0, i1, i2] -> [(i0)] }]"
        options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S13[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S13[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
    - filter: "[M, N, O, P, Q, R, S, T, U] -> { S14[i0, i1, i2] }"
      child:
        schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S14[i0, i1, i2] -> [(i0)] }]"
        options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S14[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S14[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
    - filter: "[M, N, O, P, Q, R, S, T, U] -> { S15[i0, i1, i2] }"
      child:
        schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S15[i0, i1, i2] -> [(i0)] }]"
        options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
        child:
          schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S15[i0, i1, i2] -> [(i1)] }]"
          options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
          child:
            schedule: "[M, N, O, P, Q, R, S, T, U] -> [{ S15[i0, i1, i2] -> [(i2)] }]"
            options: "[M, N, O, P, Q, R, S, T, U] -> { separate[i0] }"
