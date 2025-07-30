domain: "[M, N] -> { S2[i0, i1, 1 + i1, 99 + 100i1, i4] : i4 >= 3 and i4 >= -193 - 200i1 and i4 >= -194 + 100i0 - 200i1 and 100i0 >= -284 - 3N and i4 <= -1 + N and i4 <= -201 + 2M + N - 200i1 and i4 <= -95 + 100i0 - 200i1 and 100i0 >= -94 - N and 50i0 >= -45 - N and 3N >= -134 - M and i1 >= 0 and N >= 4 and 200i1 >= -192 - N and 200i1 >= -193 - N + 100i0 and 100i0 <= -7 + 2M + N and 7N >= -463 - 2M and 100i1 <= -100 + M and i0 >= 0 and 200i1 <= -204 + 2M + N and 2i1 <= -1 + i0 and 5N >= -75 - 2M and N >= 8 - 2M and 50i0 <= -6 + M + N and 50i0 <= 89 + M + 2N and 100i0 <= -15 + 2M + 3N and M >= 2 and 100i1 <= -5 + M + N and 2N >= -39 - M and 200i1 <= 96 + N + 100i0 and 3N >= 16 - 2M and 100i1 >= -94 - N + 50i0 and N >= 6 - M and 100i1 >= -94 - N; S3[i0, i1, 1 + i1, 99 + 100i1, i4] : i4 >= 3 and i4 >= -193 - 200i1 and i4 >= -194 + 100i0 - 200i1 and 100i0 >= -284 - 3N and i4 <= -1 + N and i4 <= -201 + 2M + N - 200i1 and i4 <= -95 + 100i0 - 200i1 and 100i0 >= -94 - N and 50i0 >= -45 - N and 3N >= -134 - M and i1 >= 0 and N >= 4 and 200i1 >= -192 - N and 200i1 >= -193 - N + 100i0 and 100i0 <= -7 + 2M + N and 7N >= -463 - 2M and 100i1 <= -100 + M and i0 >= 0 and 200i1 <= -204 + 2M + N and 2i1 <= -1 + i0 and 5N >= -75 - 2M and N >= 8 - 2M and 50i0 <= -6 + M + N and 50i0 <= 89 + M + 2N and 100i0 <= -15 + 2M + 3N and M >= 2 and 100i1 <= -5 + M + N and 2N >= -39 - M and 200i1 <= 96 + N + 100i0 and 3N >= 16 - 2M and 100i1 >= -94 - N + 50i0 and N >= 6 - M and 100i1 >= -94 - N; S4[i0] : 200i0 >= -781 - 3N and 200i0 >= -391 - N and 50i0 >= -268 - N and 100i0 >= -392 - N and i0 >= -1 and 200i0 <= 377 + 6M + 5N and 100i0 <= 335 + 3M + 3N and 100i0 <= 190 + 3M + 2N and 200i0 <= -13 + 6M + 3N and 100i0 <= -5 + 3M + N and 3N >= -484 - 2M and N >= -95 - M and N >= -192 - 3M and 5N >= -873 - 3M and 2N >= -189 - 3M and 7N >= -1062 - 6M and 5N >= -771 - 6M and 4N >= -579 - 3M and N >= 3 and N >= 5 - 2M and M >= 1; S1[i0, i1, i2, i3] : i3 >= 4 + 100i0 - 2i2 and i3 >= 2 and i3 <= 103 + 100i0 - 2i2 and i3 <= -1 + N and i2 >= 1 and i2 >= 100i1 and 2i2 >= 5 - N + 100i0 and i2 <= M and i2 <= 99 + 100i1 and i2 <= 50 + 50i0 and i1 >= 0 and 200i1 >= -193 - N + 100i0 and 100i1 <= M and 2i1 <= 1 + i0 and i0 >= 0 and 100i0 <= -5 + 2M + N and N >= 3 and N >= -94 - 2M and M >= 1 }"
child:
  context: "[M, N] -> { [] : M >= 0 and N >= 0 }"
  child:
    schedule: "[M, N] -> [{ S2[i0, i1, i2, i3, i4] -> [(i0 + i1)]; S1[i0, i1, i2, i3] -> [(i0 + i1)]; S3[i0, i1, i2, i3, i4] -> [(1 + i0 + i1)]; S4[i0] -> [(i0)] }]"
    options: "[M, N] -> { atomic[i0] }"
    child:
      sequence:
      - filter: "[M, N] -> { S2[i0, i1, i2, i3, i4]; S3[i0, i1, i2, i3, i4]; S1[i0, i1, i2, i3] }"
        child:
          schedule: "[M, N] -> [{ S2[i0, i1, i2, i3, i4] -> [(i1)]; S1[i0, i1, i2, i3] -> [(i1)]; S3[i0, i1, i2, i3, i4] -> [(i2)] }, { S2[i0, i1, i2, i3, i4] -> [(-4 + 2i3 + i4)]; S1[i0, i1, i2, i3] -> [(-4 + 2i2 + i3)]; S3[i0, i1, i2, i3, i4] -> [(-4 + 2i3 + i4)] }, { S2[i0, i1, i2, i3, i4] -> [(i3)]; S1[i0, i1, i2, i3] -> [(i2)]; S3[i0, i1, i2, i3, i4] -> [(i3)] }]"
          options: "[M, N] -> { atomic[i0] }"
          child:
            sequence:
            - filter: "[M, N] -> { S3[i0, i1, i2, i3, i4] }"
              child:
                schedule: "[M, N] -> [{ S3[i0, i1, i2, i3, i4] -> [(i1)] }, { S3[i0, i1, i2, i3, i4] -> [(i4)] }]"
                options: "[M, N] -> { atomic[i0] }"
            - filter: "[M, N] -> { S1[i0, i1, i2, i3] }"
            - filter: "[M, N] -> { S2[i0, i1, i2, i3, i4] }"
              child:
                schedule: "[M, N] -> [{ S2[i0, i1, i2, i3, i4] -> [(i2)] }, { S2[i0, i1, i2, i3, i4] -> [(i4)] }]"
                options: "[M, N] -> { atomic[i0] }"
      - filter: "[M, N] -> { S4[i0] }"
