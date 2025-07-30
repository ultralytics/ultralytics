domain: "[n] -> { S2[i, j] : 29j >= 1 - i and i <= n and j >= 1 and j <= -1 + i; S1[i] : i >= 1 - 27n and i <= 28 + n; S4[i, j] : i >= 1 and i <= n and j >= 1 + i and j <= n; S5[i, j, k] : i >= 1 and i <= n and j >= 1 + i and j <= n and k >= 1 and k <= -1 + i; S6[i, j] : i >= 1 and i <= n and j >= 1 + i and j <= n; S3[i] : i >= 1 and i <= n }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S1[i0] -> [(2 + 2i0)]; S4[i0, i1] -> [(2i0 + 2i1)]; S6[i0, i1] -> [(2i0 + 2i1)]; S3[i0] -> [(1 + 4i0)]; S5[i0, i1, i2] -> [(2i0 + 2i1)]; S2[i0, i1] -> [(1 + 2i0 + 2i1)] }, { S1[i0] -> [(0)]; S4[i0, i1] -> [(-i0)]; S6[i0, i1] -> [(2 - i0)]; S3[i0] -> [(0)]; S5[i0, i1, i2] -> [(1 - i0)]; S2[i0, i1] -> [(i1)] }, { S1[i0] -> [(0)]; S4[i0, i1] -> [(0)]; S6[i0, i1] -> [(0)]; S3[i0] -> [(0)]; S5[i0, i1, i2] -> [(i2)]; S2[i0, i1] -> [(0)] }]"
    options: "[n] -> { separate[i0] }"
